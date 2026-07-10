from pathlib import Path
import streamlit as st
import helper
import settings
import json
import cv2
import numpy as np
from PIL import Image
import folium
import pandas as pd
from datetime import datetime
from recycling_center_helper import find_recycling_centers, get_all_nearby_facilities
from geolocation_helper import get_user_location, get_address_from_coordinates
from streamlit.components.v1 import html as st_html

# MUST BE FIRST UI CALL
st.set_page_config(
    page_title="EcoVision AI - Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Init
for key, val in {
    'latest_detection': None,
    'frozen_object': None,
    'unique_classes': set(),
    'camera_running': False,
    'user_lat': 28.6080,
    'user_lon': 77.4580,
    'captured_objects': [],
    'conf_threshold': 0.3,
    'chat_history': []
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background: #f8fafc; padding: 1.5rem; }
    
    /* Header Card styling */
    .header-container { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
        padding: 2.5rem; 
        border-radius: 24px; 
        color: white; 
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15); 
        margin-bottom: 2rem; 
        position: relative;
        overflow: hidden;
    }
    .header-container::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-title { font-size: 2.75rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: -1px; background: linear-gradient(to right, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { font-size: 1.1rem; color: #94a3b8; margin-top: 0.5rem; font-weight: 300; }
    
    /* Card design */
    .custom-card {
        background: white;
        padding: 1.75rem;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #f1f5f9;
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Stat badges */
    .stat-card {
        background: white;
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid #f1f5f9;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08);
    }
    .custom-card {
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .custom-card:hover {
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08);
    }
    .stat-val { font-size: 1.75rem; font-weight: 700; color: #0f172a; }
    .stat-lbl { font-size: 0.8rem; color: #64748b; font-weight: 500; text-transform: uppercase; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-container">
    <div class="main-title">♻️ EcoVision AI</div>
    <div class="subtitle">Next-Generation Waste Segregation & Environmental Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

# Load model
model_path = Path(settings.DETECTION_MODEL)
model = None
try:
    model = helper.load_model(model_path)
except Exception as e:
    st.error(f"Error loading YOLOv8 model: {e}")

# Sidebar controls
st.sidebar.image("https://img.icons8.com/external-flat-icons-inmotus-design/150/external-Ecology-recycling-flat-icons-inmotus-design.png", width=90)
st.sidebar.title("Configuration")

# 1. Choose Input Source
input_mode = st.sidebar.selectbox(
    "Select Input Source",
    ["📸 Take Photo (Browser Camera)", "📤 Upload Image/Video", "📹 Live Stream (Local OpenCV)"]
)

# Confidence slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, float(st.session_state["conf_threshold"]), 0.05
)
st.session_state["conf_threshold"] = conf_threshold

# 2. Manual Waste Entry Section
with st.sidebar.expander("➕ Log Waste Manually", expanded=False):
    with st.form("manual_log_form"):
        m_name = st.text_input("Item Name", value="Plastic Bottle")
        m_cat = st.selectbox("Category", ["Recyclable", "Non-Recyclable", "Hazardous"])
        m_qty = st.number_input("Quantity", min_value=1, value=1)
        m_notes = st.text_input("Notes", value="Manually logged")
        m_submit = st.form_submit_button("Log Item")
        if m_submit:
            norm_cls = m_name.lower().replace(" ", "_")
            resolved_cat = m_cat
            if norm_cls in settings.RECYCLABLE:
                resolved_cat = "Recyclable"
            elif norm_cls in settings.HAZARDOUS:
                resolved_cat = "Hazardous"
            elif norm_cls in settings.NON_RECYCLABLE:
                resolved_cat = "Non-Recyclable"
                
            rec_key = settings.CLASS_TO_REC_KEY.get(norm_cls, 'plastic' if resolved_cat == "Recyclable" else 'non_recyclable')
            impact = settings.IMPACT_FACTORS.get(rec_key, {'co2': 0, 'water': 0, 'energy': 0})
            st.session_state["captured_objects"].append({
                "object": m_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": resolved_cat,
                "quantity": m_qty,
                "notes": m_notes,
                "co2_saved": round(impact['co2'] * m_qty, 3),
                "water_saved": round(impact['water'] * m_qty, 3),
                "energy_saved": round(impact['energy'] * m_qty, 3)
            })
            st.success(f"Logged successfully as {resolved_cat}!")
            st.rerun()

# 3. Smart Eco Chatbot Assistant
with st.sidebar.expander("🤖 Eco AI Assistant", expanded=True):
    st.caption("Ask questions about recycling or waste management:")
    
    # Pre-defined buttons for quick questions
    q_cols = st.columns(2)
    with q_cols[0]:
        if st.button("Pizza Box?", key="q_pizza"):
            query = "pizza box"
            st.session_state["chat_history"].append(("user", query))
            st.session_state["chat_history"].append(("assistant", settings.ECO_KNOWLEDGE_BASE["pizza box"]))
    with q_cols[1]:
        if st.button("Batteries?", key="q_battery"):
            query = "battery"
            st.session_state["chat_history"].append(("user", query))
            st.session_state["chat_history"].append(("assistant", settings.ECO_KNOWLEDGE_BASE["battery"]))
            
    # Chat output using native components
    chat_container = st.container(height=220)
    with chat_container:
        for role, text in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(text)
                
    # Chat input at bottom
    chat_input = st.chat_input("Ask about recycling...")
    if chat_input:
        query = chat_input.strip().lower()
        st.session_state["chat_history"].append(("user", chat_input))
        
        # 1. Check exact or partial match in ECO_KNOWLEDGE_BASE
        reply = None
        for k, val in settings.ECO_KNOWLEDGE_BASE.items():
            if k in query or query in k:
                reply = val
                break
                
        # 2. If not found, check against YOLO classes
        if not reply:
            matched_cls = None
            clean_query = query.replace(" ", "_").replace("-", "_")
            for cls_name in list(settings.CLASS_TO_REC_KEY.keys()):
                if cls_name in clean_query or clean_query in cls_name:
                    matched_cls = cls_name
                    break
            
            if matched_cls:
                rec_key = settings.CLASS_TO_REC_KEY[matched_cls]
                rec_text = settings.RECOMMENDATIONS.get(rec_key, "Dispose of responsibly in the general waste bin.")
                
                category = "Non-Recyclable"
                if matched_cls in settings.RECYCLABLE:
                    category = "Recyclable"
                elif matched_cls in settings.HAZARDOUS:
                    category = "Hazardous"
                
                reply = f"**{matched_cls.replace('_', ' ').title()}** is classified as **{category}**.\n\n*Recommendation*: {rec_text}"
        
        # 3. Fallback generic reply
        if not reply:
            reply = "I'm not fully sure about that item. Generally, if it is clean paper, plastic bottle, metal, or glass, it is recyclable. If dirty, contaminated, or organic, dispose of it in general waste or compost."
            
        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()
            
    if st.button("Clear Chat", key="clear_chat_history"):
        st.session_state["chat_history"] = []
        st.rerun()

# Main layout cols
col1, col2 = st.columns([1.6, 1.0])

def render_log_form(detected_cls, form_key):
    with st.expander("📝 Edit and Log item to History", expanded=True):
        with st.form(form_key):
            norm_cls = detected_cls.lower().replace(" ", "_")
            default_cat = "Recyclable" if norm_cls in settings.RECYCLABLE else ("Hazardous" if norm_cls in settings.HAZARDOUS else "Non-Recyclable")
            
            edit_name = st.text_input("Item Name", value=helper.remove_dash_from_class_name(detected_cls))
            edit_cat = st.selectbox("Category", ["Recyclable", "Non-Recyclable", "Hazardous"], index=["Recyclable", "Non-Recyclable", "Hazardous"].index(default_cat))
            edit_qty = st.number_input("Quantity / Count", min_value=1, value=1)
            edit_notes = st.text_input("Notes (e.g., Cleaned, sorted)", value="Cleaned & ready")
            
            submit_log = st.form_submit_button("💾 Save Item to History Dashboard")
            if submit_log:
                rec_key = settings.CLASS_TO_REC_KEY.get(norm_cls, 'non_recyclable')
                impact = settings.IMPACT_FACTORS.get(rec_key, {'co2': 0, 'water': 0, 'energy': 0})
                st.session_state["captured_objects"].append({
                    "object": edit_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": edit_cat,
                    "quantity": edit_qty,
                    "notes": edit_notes,
                    "co2_saved": round(impact['co2'] * edit_qty, 3),
                    "water_saved": round(impact['water'] * edit_qty, 3),
                    "energy_saved": round(impact['energy'] * edit_qty, 3)
                })
                st.success("Logged successfully!")
                st.rerun()

def render_results(obj_name, placeholder):
    if not obj_name:
        placeholder.empty()
        return

    norm_name = obj_name.lower().replace(" ", "_")
    rec_key = settings.CLASS_TO_REC_KEY.get(norm_name)
    recommendation = settings.RECOMMENDATIONS.get(rec_key, "Dispose of responsibly in the general waste bin.")
    lat, lon = st.session_state.get("user_lat"), st.session_state.get("user_lon")
    
    # Determine category for badge
    category = "Unknown"
    color = "#64748b"
    if norm_name in settings.RECYCLABLE:
        category = "Recyclable"
        color = "#10b981"
    elif norm_name in settings.NON_RECYCLABLE:
        category = "Non-Recyclable"
        color = "#ef4444"
    elif norm_name in settings.HAZARDOUS:
        category = "Hazardous"
        color = "#f59e0b"

    html_out = ""
    if recommendation:
        html_out += f'<div style="background: white; padding: 1.5rem; border-radius: 20px; border: 1px solid #f1f5f9; margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.02); position: relative; overflow: hidden;">'
        html_out += f'<div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: {color};"></div>'
        html_out += f'<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">'
        html_out += f'<div><div style="font-weight: 700; color: #0f172a; font-size: 1.3rem; margin-bottom: 2px;">{obj_name.replace("_", " ").title()}</div>'
        html_out += f'<span style="display: inline-block; padding: 3px 12px; border-radius: 20px; background: {color}20; color: {color}; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">{category}</span></div>'
        html_out += f'<div style="background: #f8fafc; width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">💡</div></div>'
        html_out += f'<div style="color: #334155; font-size: 0.95rem; line-height: 1.7; background: #f8fafc; padding: 1rem; border-radius: 12px; border: 1px solid #f1f5f9;">{recommendation}</div></div>'
    
    if lat and lon:
        address = get_address_from_coordinates(lat, lon)
        html_out += f'<div style="background: #f8fafc; padding: 1.25rem; border-radius: 16px; border: 1px solid #e2e8f0; margin-bottom: 1.5rem; display: flex; align-items: center;">'
        html_out += f'<div style="margin-right: 15px; font-size: 1.5rem;">📍</div>'
        html_out += f'<div><div style="font-weight: 700; color: #475569; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">Current Address Reference</div>'
        html_out += f'<div style="color: #64748b; font-size: 0.9rem;">{address}</div></div></div>'
        
        centers = find_recycling_centers(obj_name, lat, lon)
        if centers:
            html_out += '<h3 style="color:#0f172a; margin-bottom:1rem; font-size:1.15rem; font-weight:700; display:flex; align-items:center; gap:8px;">♻️ <span>Nearby Specialized Centers</span></h3>'
            for c in centers:
                maps_url = f"https://www.google.com/maps/search/?api=1&query={c['latitude']},{c['longitude']}"
                badges = "".join([f'<span style="display:inline-block; background:#f0f9ff; color:#0284c7; font-size:0.7rem; padding:3px 8px; border-radius:6px; margin-right:6px; margin-bottom:6px; font-weight:600;">{w}</span>' for w in c["accepted_waste"][:3]])
                html_out += f'<div style="background:white; padding:1.25rem; margin-bottom:12px; border-radius:16px; border:1px solid #f1f5f9; transition: all 0.2s ease;">'
                html_out += f'<a href="{maps_url}" target="_blank" style="text-decoration:none; color:inherit;">'
                html_out += f'<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">'
                html_out += f'<div style="font-weight:700; color:#0f172a; font-size:1rem; flex:1;">{c["name"]}</div>'
                html_out += f'<div style="color:#10b981; font-weight:700; font-size:0.9rem; background:#ecfdf5; padding:2px 8px; border-radius:8px; white-space:nowrap; margin-left:10px;">{c["distance_km"]} km</div></div>'
                html_out += f'<div style="margin-bottom:12px; display:flex; flex-wrap:wrap;">{badges}</div>'
                html_out += f'<div style="display:flex; align-items:center; justify-content:space-between; color:#64748b; font-size:0.8rem;">'
                html_out += f'<span>⏰ {c["hours"]}</span>'
                html_out += f'<span style="color:#0284c7; font-weight:700;">Get Directions →</span></div></a></div>'
    
    placeholder.markdown(html_out, unsafe_allow_html=True)

with col1:
    # Render main interaction zone based on input source
    if input_mode == "📸 Take Photo (Browser Camera)":
        st.markdown("""
        <div class="custom-card">
            <h3 class="card-title">📸 Device Camera Feed</h3>
            <p style="color:#64748b; font-size:0.9rem; margin-bottom:1rem;">Capture an image of a waste item using your browser's webcam or phone camera.</p>
        </div>
        """, unsafe_allow_html=True)
        
        img_file = st.camera_input("Snap a picture to segregate:")
        result_placeholder = st.empty()
        
        if img_file is not None and model is not None:
            image = Image.open(img_file)
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            plotted_bgr, detected_cls, all_detected = helper.predict_single_image(model, img_bgr, conf_threshold)
            plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
            
            st.image(plotted_rgb, caption="Analyzed Image", use_container_width=True)
            
            if detected_cls:
                st.session_state['latest_detection'] = detected_cls
                render_results(detected_cls, result_placeholder)
                render_log_form(detected_cls, "log_form_browser_cam")
            else:
                st.info("No items detected. Try adjustments to confidence threshold or light/angle.")
                
    elif input_mode == "📤 Upload Image/Video":
        st.markdown("""
        <div class="custom-card">
            <h3 class="card-title">📤 Upload File (Image or Video)</h3>
            <p style="color:#64748b; font-size:0.9rem; margin-bottom:1rem;">Upload a JPEG/PNG image or an MP4 video for waste detection analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose file...", type=["jpg", "jpeg", "png", "mp4"])
        result_placeholder = st.empty()
        
        if uploaded_file is not None and model is not None:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext in ['jpg', 'jpeg', 'png']:
                image = Image.open(uploaded_file)
                img_np = np.array(image)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                plotted_bgr, detected_cls, all_detected = helper.predict_single_image(model, img_bgr, conf_threshold)
                plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
                
                st.image(plotted_rgb, caption="Uploaded Image Analysis", use_container_width=True)
                
                if detected_cls:
                    st.session_state['latest_detection'] = detected_cls
                    render_results(detected_cls, result_placeholder)
                    render_log_form(detected_cls, "log_form_upload_img")
                else:
                    st.info("No items detected.")
            else:
                st.info("Processing video file...")
                temp_path = Path("temp_video.mp4")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                    
                vid = cv2.VideoCapture(str(temp_path))
                try:
                    st_frame = st.empty()
                    
                    while vid.isOpened():
                        ret, frame = vid.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, (640, 480))
                        
                        res = model.predict(frame, conf=conf_threshold, verbose=False)
                        names = model.names
                        
                        if len(res[0].boxes) > 0:
                            top_box = sorted(res[0].boxes, key=lambda x: float(x.conf[0]), reverse=True)[0]
                            cls_idx = int(top_box.cls[0])
                            detected_cls = names[cls_idx]
                            st.session_state['latest_detection'] = detected_cls
                            render_results(detected_cls, result_placeholder)
                            
                        res_plotted = res[0].plot()
                        st_frame.image(res_plotted, channels="BGR")
                finally:
                    vid.release()
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass
                st.success("Video processing finished.")
                
    elif input_mode == "📹 Live Stream (Local OpenCV)":
        st.markdown("""
        <div class="custom-card">
            <h3 class="card-title">📹 Host Machine Camera (OpenCV)</h3>
            <p style="color:#64748b; font-size:0.9rem; margin-bottom:1rem;">Note: This mode streams directly from the host system's hardware webcam (port 0).</p>
        </div>
        """, unsafe_allow_html=True)
        
        result_placeholder = st.empty()
        
        if st.session_state.get('frozen_object'):
            st.info("🔒 Detection paused.")
            if st.button("🔄 Clear Captured Object"):
                st.session_state['frozen_object'] = None
                st.session_state['latest_detection'] = None
                st.rerun()
        else:
            if model is not None:
                helper.play_webcam(model, result_placeholder, render_results)

with col2:
    st.markdown("""
    <div class="custom-card">
        <h3 class="card-title">📍 Geolocation & Mapping</h3>
    </div>
    """, unsafe_allow_html=True)
    
    lat, lon = st.session_state["user_lat"], st.session_state["user_lon"]
    address = get_address_from_coordinates(lat, lon)
    st.caption(f"Coordinates: {lat:.6f}, {lon:.6f}")
    st.info(f"📍 {address}")
    
    try:
        m = folium.Map(location=[lat, lon], zoom_start=13, control_scale=True)
        folium.Marker(
            [lat, lon],
            popup="Your Location",
            tooltip="You are here",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
        
        obj = st.session_state.get('latest_detection')
        if obj:
            centers = find_recycling_centers(obj, lat, lon)
        else:
            centers = get_all_nearby_facilities(lat, lon, top_k=5)
        
        for c in centers:
            folium.Marker(
                [c['latitude'], c['longitude']],
                popup=f"<b>{c['name']}</b><br>Distance: {c['distance_km']} km<br>Hours: {c['hours']}",
                tooltip=c['name'],
                icon=folium.Icon(color="green", icon="leaf")
            ).add_to(m)
            
        map_html = m._repr_html_()
        st_html(map_html, height=280)
    except Exception as map_err:
        st.write(f"Map load error: {map_err}")

    with st.expander("🌐 Update GPS Location Reference"):
        new_lat = st.number_input("Latitude", value=float(lat), format="%.6f")
        new_lon = st.number_input("Longitude", value=float(lon), format="%.6f")
        if st.button("Set Coordinates Manually"):
            st.session_state["user_lat"] = new_lat
            st.session_state["user_lon"] = new_lon
            st.success("Coordinates updated!")
            st.rerun()
            
        if st.button("🛰️ Detect Live Location (GPS/IP)"):
            with st.spinner("Locating..."):
                det_lat, det_lon = get_user_location()
                st.session_state["user_lat"] = det_lat
                st.session_state["user_lon"] = det_lon
                st.success("Updated Location Reference!")
                st.rerun()

    st.markdown("""
    <div class="custom-card">
        <h3 class="card-title">📖 Waste Categories</h3>
        <div style="font-size:0.85rem; margin-bottom:8px;"><strong style="color:#10b981;">♻️ Recyclable:</strong> Paper, cardboard, glass, metals, clean plastic bottles.</div>
        <div style="font-size:0.85rem; margin-bottom:8px;"><strong style="color:#ef4444;">🗑️ Non-Recyclable:</strong> Tissues, plastic films, food contaminated boxes, Styrofoam, shoes.</div>
        <div style="font-size:0.85rem;"><strong style="color:#f59e0b;">⚠️ Hazardous:</strong> Batteries, light bulbs, paint buckets, chemical containers.</div>
    </div>
    """, unsafe_allow_html=True)

# 3. Bottom Dashboard (Analytics & Log History)
st.markdown("---")
st.markdown("## 📊 Environmental Analytics & Savings Dashboard")

history = st.session_state["captured_objects"]
total_scans = len(history)

rec_count = sum(1 for item in history if item.get('category') == 'Recyclable')
non_rec_count = sum(1 for item in history if item.get('category') == 'Non-Recyclable')
haz_count = sum(1 for item in history if item.get('category') == 'Hazardous')

# Calculate ecological savings
total_co2 = sum(item.get('co2_saved', 0.0) for item in history)
total_water = sum(item.get('water_saved', 0.0) for item in history)
total_energy = sum(item.get('energy_saved', 0.0) for item in history)

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val">{total_scans}</div>
        <div class="stat-lbl">Total Scans Logged</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val" style="color:#10b981;">{rec_count}</div>
        <div class="stat-lbl">Recyclable Items</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val" style="color:#ef4444;">{non_rec_count}</div>
        <div class="stat-lbl">Non-Recyclable Items</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val" style="color:#f59e0b;">{haz_count}</div>
        <div class="stat-lbl">Hazardous Items</div>
    </div>
    """, unsafe_allow_html=True)

with m_col5:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-val" style="color:#0284c7;">{len(st.session_state["unique_classes"])}</div>
        <div class="stat-lbl">Unique Waste Types</div>
    </div>
    """, unsafe_allow_html=True)

# Ecological Savings Cards
st.markdown("### 🍃 Resource Savings Tracker")
s_col1, s_col2, s_col3 = st.columns(3)
with s_col1:
    st.markdown(f"""
    <div class="stat-card" style="border-top: 4px solid #10b981; margin-bottom: 1.5rem;">
        <div class="stat-val" style="color:#10b981;">{total_co2:.2f} kg</div>
        <div class="stat-lbl">CO₂ Emissions Avoided</div>
    </div>
    """, unsafe_allow_html=True)
with s_col2:
    st.markdown(f"""
    <div class="stat-card" style="border-top: 4px solid #0284c7; margin-bottom: 1.5rem;">
        <div class="stat-val" style="color:#0284c7;">{total_water:.1f} L</div>
        <div class="stat-lbl">Water Resources Saved</div>
    </div>
    """, unsafe_allow_html=True)
with s_col3:
    st.markdown(f"""
    <div class="stat-card" style="border-top: 4px solid #f59e0b; margin-bottom: 1.5rem;">
        <div class="stat-val" style="color:#f59e0b;">{total_energy:.1f} kWh</div>
        <div class="stat-lbl">Energy Saved</div>
    </div>
    """, unsafe_allow_html=True)

db_col1, db_col2 = st.columns([1, 1])

with db_col1:
    st.markdown("### 📈 Segregation Distribution")
    if total_scans > 0:
        chart_data = pd.DataFrame({
            'Category': ['Recyclable', 'Non-Recyclable', 'Hazardous'],
            'Count': [rec_count, non_rec_count, haz_count]
        }).set_index('Category')
        st.bar_chart(chart_data)
    else:
        st.info("Log scanned items to see real-time chart data.")

with db_col2:
    st.markdown("### 📋 Captured Logs History")
    if total_scans > 0:
        st.caption(f"🕐 Last scan logged at {history[-1]['timestamp']}")
        log_df = pd.DataFrame(history)
        st.dataframe(log_df.iloc[::-1], use_container_width=True)
        st.download_button(
            "⬇️ Export Logs as CSV",
            log_df.to_csv(index=False),
            file_name=f"waste_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        if st.button("🧹 Clear History Logs"):
            st.session_state["captured_objects"] = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No items have been logged in the current session.")

st.markdown("""
<div style="margin-top: 3rem; text-align: center; color: #94a3b8; font-size: 0.8rem; border-top: 1px solid #e2e8f0; padding-top: 1.5rem; padding-bottom: 1.5rem;">
    EcoVision AI v2.1 • Intelligent Segregation Dashboard • Designed with premium clean UI aesthetics
</div>
""", unsafe_allow_html=True)