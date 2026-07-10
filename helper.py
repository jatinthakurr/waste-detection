from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading
from datetime import datetime

# -------------------------------------------
# LOAD MODEL
# -------------------------------------------
def load_model(model_path):
    # Register safe globals for PyTorch 2.6+ compatibility
    import torch
    original_load = torch.load
    try:
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
        model = YOLO(model_path)
    finally:
        torch.load = original_load
    return model

# -------------------------------------------
# WASTE CLASSIFICATION helper
# -------------------------------------------
def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    return recyclable_items, non_recyclable_items, hazardous_items

def remove_dash_from_class_name(class_name):
    if not class_name:
        return ""
    return class_name.replace("_", " ").replace("-", " ").title()

# -------------------------------------------
# DISPLAY DETECTION
# -------------------------------------------
def predict_single_image(model, image, conf_threshold=0.3):
    """
    Predict on a single image (numpy array or PIL Image).
    Returns (plotted_image_bgr, detected_class, all_detected_classes)
    """
    res = model.predict(image, conf=conf_threshold, verbose=False)
    names = model.names
    detected_class = None
    all_detected_classes = []
    
    for result in res:
        if len(result.boxes) > 0:
            top_box = sorted(result.boxes, key=lambda x: float(x.conf[0]), reverse=True)[0]
            cls_idx = int(top_box.cls[0])
            detected_class = names[cls_idx]
            
            for box in result.boxes:
                cls_name = names[int(box.cls[0])]
                all_detected_classes.append(cls_name)
                if 'unique_classes' in st.session_state:
                    st.session_state['unique_classes'].add(cls_name)
                    
    res_plotted = res[0].plot()
    return res_plotted, detected_class, all_detected_classes

def _display_detected_frames(model, st_frame, image, enable_prediction=True):
    if "no_det_frames" not in st.session_state:
        st.session_state["no_det_frames"] = 0


    if not enable_prediction:
        st_frame.image(image, channels="BGR")
        return

    conf = st.session_state.get('conf_threshold', 0.3)
    res = model.predict(image, conf=conf, verbose=False)

    if "latest_detection" not in st.session_state:
        st.session_state["latest_detection"] = None

    names = model.names
    
    # Track current frame detections
    current_frame_classes = set()
    
    for result in res:
        if len(result.boxes) > 0:
            # Sort by confidence and pick the top one for the recommendation card
            top_box = sorted(result.boxes, key=lambda x: x.conf, reverse=True)[0]
            cls_idx = int(top_box.cls[0])
            detected_class = names[cls_idx]
            
            st.session_state["latest_detection"] = detected_class
            st.session_state["no_det_frames"] = 0
            
            # For unique class tracking (sidebar history)
            for box in result.boxes:
                st.session_state['unique_classes'].add(names[int(box.cls[0])])
        else:
            st.session_state["no_det_frames"] += 1
            if st.session_state["no_det_frames"] > 5:
                st.session_state["latest_detection"] = None

    # Show plotted frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, channels="BGR")

# -------------------------------------------
# MAIN WEBCAM FUNCTION
# -------------------------------------------
def play_webcam(model, result_placeholder=None, render_results=None):
    source_webcam = settings.WEBCAM_PATH
    
    # -------------------------------------
    # Sidebar Info
    # -------------------------------------
    st.sidebar.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.3, 0.05
    )
    st.session_state["conf_threshold"] = conf_threshold

    with st.sidebar.expander("📋 Detectable Objects"):
        for idx, name in model.names.items():
            st.sidebar.text(f"{idx}: {name}")

    # Initialize State
    if "captured_objects" not in st.session_state:
        st.session_state["captured_objects"] = []
    if "frame_count" not in st.session_state:
        st.session_state["frame_count"] = 0
    if "latest_detection" not in st.session_state:
        st.session_state["latest_detection"] = None
    if "stable_detection" not in st.session_state:
        st.session_state["stable_detection"] = None
    if "stable_detection_count" not in st.session_state:
        st.session_state["stable_detection_count"] = 0
    if "unique_classes" not in st.session_state:
        st.session_state["unique_classes"] = set()

    # History in sidebar
    if st.session_state["captured_objects"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Captured History")
        for obj in reversed(st.session_state["captured_objects"][-5:]):
            st.sidebar.text(f"• {obj['object']}")

    # Control Buttons
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start Detection", use_container_width=True)
        if start_btn:
            st.session_state["camera_running"] = True
    with col2:
        stop_btn = st.button("⏹️ Stop Detection", use_container_width=True)
        if stop_btn:
            st.session_state["camera_running"] = False

    # -------------------------------------
    # CAMERA LOOP
    # -------------------------------------
    if st.session_state.get("camera_running"):
        vid = cv2.VideoCapture(source_webcam)
        try:
            st_frame = st.empty()
            status_info = st.empty()
            capture_placeholder = st.sidebar.empty()
            
            while st.session_state.get("camera_running"):
                ok, frame = vid.read()
                if not ok:
                    st.error("Camera connection failed.")
                    break

                st.session_state["frame_count"] += 1
                _display_detected_frames(model, st_frame, frame)

                # Stability Logic
                detected = st.session_state.get("latest_detection")
                if detected:
                    if detected == st.session_state.get("stable_detection"):
                        st.session_state["stable_detection_count"] += 1
                    else:
                        st.session_state["stable_detection"] = detected
                        st.session_state["stable_detection_count"] = 1
                else:
                    if st.session_state["stable_detection_count"] > 0:
                        st.session_state["stable_detection_count"] -= 1
                    else:
                        st.session_state["stable_detection"] = None
                
                # UI Updates for stable detections
                stable_obj = st.session_state.get("stable_detection")
                if stable_obj and st.session_state["stable_detection_count"] >= 3:
                    # Update main recommendation card
                    if result_placeholder and render_results:
                        render_results(stable_obj, result_placeholder)
                    
                    # Show Capture Button in sidebar
                    btn_key = f"cap_{st.session_state['frame_count']}"
                    if capture_placeholder.button(f"📸 Capture {remove_dash_from_class_name(stable_obj)}", key=btn_key):
                        norm_cls = stable_obj.lower().replace(" ", "_")
                        cat = "Recyclable" if norm_cls in settings.RECYCLABLE else ("Hazardous" if norm_cls in settings.HAZARDOUS else "Non-Recyclable")
                        rec_key = settings.CLASS_TO_REC_KEY.get(norm_cls, 'non_recyclable')
                        impact = settings.IMPACT_FACTORS.get(rec_key, {'co2': 0, 'water': 0, 'energy': 0})
                        st.session_state["captured_objects"].append({
                            "object": remove_dash_from_class_name(stable_obj),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "category": cat,
                            "quantity": 1,
                            "notes": "Captured via live webcam feed",
                            "co2_saved": round(impact['co2'], 3),
                            "water_saved": round(impact['water'], 3),
                            "energy_saved": round(impact['energy'], 3)
                        })
                        st.session_state["frozen_object"] = stable_obj
                        st.session_state["camera_running"] = False
                        break
                else:
                    capture_placeholder.empty()
        finally:
            vid.release()
            st.session_state["camera_running"] = False
            st.rerun()