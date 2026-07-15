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

# 3. Smart Eco Chatbot Assistant (Rendered as premium floating widget at the bottom of the page)

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
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("🧹 Clear History Logs", use_container_width=True):
                st.session_state["captured_objects"] = []
                st.success("History cleared!")
                st.rerun()
        with action_cols[1]:
            if st.button("🗑️ Undo Last Log", use_container_width=True):
                if st.session_state["captured_objects"]:
                    removed = st.session_state["captured_objects"].pop()
                    st.success(f"Removed '{removed['object']}'!")
                    st.rerun()
    else:
        st.info("No items have been logged in the current session.")

st.markdown("""
<div style="margin-top: 3rem; text-align: center; color: #94a3b8; font-size: 0.8rem; border-top: 1px solid #e2e8f0; padding-top: 1.5rem; padding-bottom: 1.5rem;">
    EcoVision AI v2.1 • Intelligent Segregation Dashboard • Designed with premium clean UI aesthetics
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# PREMIUM FLOATING AI ECO-ASSISTANT WIDGET
# ----------------------------------------------------
import json

kb_json = json.dumps(settings.ECO_KNOWLEDGE_BASE)
class_rec_json = json.dumps(settings.CLASS_TO_REC_KEY)
rec_json = json.dumps(settings.RECOMMENDATIONS)
recyclable_json = json.dumps(settings.RECYCLABLE)
hazardous_json = json.dumps(settings.HAZARDOUS)

chatbot_html_template = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: transparent;
        overflow: hidden;
        width: 100%;
        height: 100%;
    }
    
    /* Floating Action Button (FAB) */
    .chat-fab {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.35);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 1000;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid rgba(255,255,255,0.15);
    }
    
    .chat-fab:hover {
        transform: scale(1.08) rotate(15deg);
        box-shadow: 0 12px 28px rgba(16, 185, 129, 0.45);
    }
    
    .chat-fab svg {
        width: 28px;
        height: 28px;
        fill: currentColor;
    }
    
    /* Pulsing effect for FAB */
    .chat-fab::after {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        border: 2px solid #10b981;
        opacity: 0.6;
        animation: pulse 2s infinite;
        z-index: -1;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.6; }
        100% { transform: scale(1.4); opacity: 0; }
    }
    
    /* Chat Container Window */
    .chat-window {
        position: fixed;
        bottom: 95px;
        right: 20px;
        width: 345px;
        height: 500px;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.6);
        display: flex;
        flex-direction: column;
        z-index: 1000;
        opacity: 0;
        transform: translateY(20px) scale(0.95);
        pointer-events: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    
    .chat-window.open {
        opacity: 1;
        transform: translateY(0) scale(1);
        pointer-events: auto;
    }
    
    /* Header */
    .chat-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 14px 18px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .chat-header-info {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .bot-avatar-header {
        width: 36px;
        height: 36px;
        border-radius: 12px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    
    .chat-header-title {
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: -0.2px;
    }
    
    .chat-header-subtitle {
        font-size: 0.72rem;
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 4px;
        margin-top: 1px;
    }
    
    .online-indicator {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 6px #10b981;
        display: inline-block;
        animation: pulse-green 1.5s infinite;
    }
    
    @keyframes pulse-green {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.5; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .close-btn {
        background: transparent;
        border: none;
        color: #94a3b8;
        cursor: pointer;
        padding: 5px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }
    
    .close-btn:hover {
        background: rgba(255,255,255,0.08);
        color: white;
    }
    
    /* Message History Area */
    .chat-messages {
        flex: 1;
        padding: 16px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 12px;
        scroll-behavior: smooth;
    }
    
    /* Custom Scrollbar */
    .chat-messages::-webkit-scrollbar {
        width: 4px;
    }
    .chat-messages::-webkit-scrollbar-track {
        background: transparent;
    }
    .chat-messages::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    /* Bubble layout */
    .message {
        display: flex;
        gap: 8px;
        max-width: 85%;
        animation: messageFadeIn 0.25s ease-out forwards;
    }
    
    @keyframes messageFadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message.user {
        align-self: flex-end;
        flex-direction: row-reverse;
    }
    
    .message.bot {
        align-self: flex-start;
    }
    
    .avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        flex-shrink: 0;
    }
    
    .user .avatar {
        background: #e0f2fe;
    }
    
    .bot .avatar {
        background: #ecfdf5;
        border: 1px solid #d1fae5;
    }
    
    .bubble {
        padding: 8px 12px;
        font-size: 0.84rem;
        line-height: 1.45;
        border-radius: 14px;
        word-break: break-word;
    }
    
    .user .bubble {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        border-radius: 14px 14px 4px 14px;
        box-shadow: 0 3px 8px rgba(2, 132, 199, 0.12);
    }
    
    .bot .bubble {
        background: white;
        color: #1e293b;
        border-radius: 14px 14px 14px 4px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.01);
    }
    
    /* Suggestion Chips */
    .suggestions-container {
        padding: 0 16px 8px 16px;
        display: flex;
        gap: 6px;
        overflow-x: auto;
        white-space: nowrap;
        scrollbar-width: none;
        flex-shrink: 0;
    }
    
    .suggestions-container::-webkit-scrollbar {
        display: none;
    }
    
    .chip {
        display: inline-block;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 5px 10px;
        font-size: 0.74rem;
        color: #475569;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.01);
        font-weight: 500;
    }
    
    .chip:hover {
        border-color: #10b981;
        color: #10b981;
        background: #f0fdf4;
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(16, 185, 129, 0.06);
    }
    
    /* Input Area */
    .chat-input-container {
        padding: 10px 16px 16px 16px;
        background: rgba(255, 255, 255, 0.5);
        border-top: 1px solid #f1f5f9;
        display: flex;
        gap: 8px;
        align-items: center;
        flex-shrink: 0;
    }
    
    .chat-input {
        flex: 1;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 8px 12px;
        font-size: 0.84rem;
        outline: none;
        background: white;
        transition: all 0.2s;
    }
    
    .chat-input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.08);
    }
    
    .send-btn {
        background: #10b981;
        border: none;
        color: white;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 3px 8px rgba(16, 185, 129, 0.15);
    }
    
    .send-btn:hover {
        background: #059669;
        transform: scale(1.02);
    }
    
    .send-btn svg {
        width: 16px;
        height: 16px;
        fill: currentColor;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        gap: 3px;
        padding: 3px 6px;
        align-items: center;
    }
    
    .typing-dot {
        width: 5px;
        height: 5px;
        background-color: #94a3b8;
        border-radius: 50%;
        animation: typingBounce 1.4s infinite ease-in-out both;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typingBounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    .clear-action {
        text-align: center;
        font-size: 0.72rem;
        color: #94a3b8;
        text-decoration: underline;
        cursor: pointer;
        padding: 2px 0;
        margin-top: -4px;
        margin-bottom: 4px;
        transition: color 0.2s;
    }
    
    .clear-action:hover {
        color: #ef4444;
    }
</style>
</head>
<body>

<div class="chat-fab" id="chatFab">
    <svg viewBox="0 0 24 24">
        <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 9h12v2H6V9zm8 5H6v-2h8v2zm4-6H6V6h12v2z"/>
    </svg>
</div>

<div class="chat-window" id="chatWindow">
    <div class="chat-header">
        <div class="chat-header-info">
            <div class="bot-avatar-header">🤖</div>
            <div>
                <div class="chat-header-title">EcoVision AI Assistant</div>
                <div class="chat-header-subtitle">
                    <span class="online-indicator"></span>
                    <span>AI Assistant • Online</span>
                </div>
            </div>
        </div>
        <button class="close-btn" id="closeBtn">
            <svg style="width:18px;height:18px;fill:currentColor;" viewBox="0 0 24 24">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
        </button>
    </div>
    
    <div class="chat-messages" id="chatMessages"></div>
    
    <div class="clear-action" id="clearBtn">Clear Conversation</div>
    
    <div class="suggestions-container" id="suggestionsContainer"></div>
    
    <div class="chat-input-container">
        <input type="text" class="chat-input" id="chatInput" placeholder="Ask about recycling..." autocomplete="off">
        <button class="send-btn" id="sendBtn">
            <svg viewBox="0 0 24 24">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
        </button>
    </div>
</div>

<script>
    const ECO_KNOWLEDGE_BASE = __ECO_KNOWLEDGE_BASE__;
    const CLASS_TO_REC_KEY = __CLASS_TO_REC_KEY__;
    const RECOMMENDATIONS = __RECOMMENDATIONS__;
    const RECYCLABLE = __RECYCLABLE__;
    const HAZARDOUS = __HAZARDOUS__;
    const SUGGESTIONS = [
        { label: "🍕 Pizza Box?", query: "pizza box" },
        { label: "🔋 Batteries?", query: "battery" },
        { label: "🌱 Compost?", query: "compost" },
        { label: "🍾 Glass Bottles?", query: "glass" },
        { label: "🥫 Soda Cans?", query: "metal" }
    ];
    
    const chatFab = document.getElementById('chatFab');
    const chatWindow = document.getElementById('chatWindow');
    const closeBtn = document.getElementById('closeBtn');
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const suggestionsContainer = document.getElementById('suggestionsContainer');
    
    let chatHistory = [];
    
    function resizeIframe(expanded) {
        const iframe = window.frameElement;
        if (iframe) {
            if (expanded) {
                iframe.style.width = '370px';
                iframe.style.height = '600px';
                iframe.style.bottom = '20px';
                iframe.style.right = '20px';
            } else {
                iframe.style.width = '90px';
                iframe.style.height = '90px';
                iframe.style.bottom = '15px';
                iframe.style.right = '15px';
            }
        }
    }
    
    function initIframeStyle() {
        const iframe = window.frameElement;
        if (iframe) {
            iframe.style.position = 'fixed';
            iframe.style.zIndex = '999999';
            iframe.style.border = 'none';
            iframe.style.background = 'transparent';
            iframe.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
            
            let parent = iframe.parentElement;
            while (parent && parent.tagName !== 'BODY') {
                parent.style.background = 'transparent';
                parent.style.border = 'none';
                parent.style.boxShadow = 'none';
                parent.style.padding = '0';
                parent.style.margin = '0';
                parent.style.overflow = 'visible';
                parent.style.pointerEvents = 'none';
                parent = parent.parentElement;
            }
            iframe.style.pointerEvents = 'auto';
            resizeIframe(false);
        }
    }
    
    function loadChatHistory() {
        const stored = localStorage.getItem('eco_chat_history');
        if (stored) {
            chatHistory = JSON.parse(stored);
        } else {
            chatHistory = [
                {
                    role: 'bot',
                    text: '👋 Welcome! Ask me anything about how to segregate or recycle waste items (e.g., "how to recycle aerosol").'
                }
            ];
        }
        renderMessages();
    }
    
    function saveChatHistory() {
        localStorage.setItem('eco_chat_history', JSON.stringify(chatHistory));
    }
    
    function renderMessages() {
        chatMessages.innerHTML = '';
        chatHistory.forEach(msg => {
            appendMessageHTML(msg.role, msg.text);
        });
        scrollToBottom();
    }
    
    function formatMarkdown(text) {
        if (!text) return "";
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }
    
    function appendMessageHTML(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.textContent = role === 'user' ? '👤' : '🤖';
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'bubble';
        bubbleDiv.innerHTML = formatMarkdown(text);
        
        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(msgDiv);
    }
    
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function renderSuggestions() {
        suggestionsContainer.innerHTML = '';
        SUGGESTIONS.forEach(s => {
            const chip = document.createElement('div');
            chip.className = 'chip';
            chip.textContent = s.label;
            chip.addEventListener('click', () => {
                handleSendMessage(s.label, s.query);
            });
            suggestionsContainer.appendChild(chip);
        });
    }
    
    chatFab.addEventListener('click', () => {
        chatWindow.classList.add('open');
        chatFab.style.transform = 'scale(0) rotate(90deg)';
        chatFab.style.opacity = '0';
        chatFab.style.pointerEvents = 'none';
        resizeIframe(true);
        setTimeout(scrollToBottom, 100);
    });
    
    closeBtn.addEventListener('click', () => {
        chatWindow.classList.remove('open');
        chatFab.style.transform = 'scale(1) rotate(0deg)';
        chatFab.style.opacity = '1';
        chatFab.style.pointerEvents = 'auto';
        setTimeout(() => resizeIframe(false), 300);
    });
    
    function processAIQuery(query) {
        let cleanQuery = query.trim().toLowerCase();
        
        // Normalize common plurals
        cleanQuery = cleanQuery
            .replace(/\bbatteries\b/g, 'battery')
            .replace(/\bbottles\b/g, 'bottle')
            .replace(/\bcans\b/g, 'can')
            .replace(/\bpapers\b/g, 'paper')
            .replace(/\bcardboards\b/g, 'cardboard')
            .replace(/\bglasses\b/g, 'glass');
        
        // 1. Check exact or partial match in ECO_KNOWLEDGE_BASE
        for (let k in ECO_KNOWLEDGE_BASE) {
            if (cleanQuery.includes(k) || k.includes(cleanQuery)) {
                return ECO_KNOWLEDGE_BASE[k];
            }
        }
        
        // 2. Check for general materials
        const materials = ['plastic', 'metal', 'glass', 'cardboard', 'paper', 'organic', 'hazardous'];
        for (let mat of materials) {
            if (cleanQuery.includes(mat)) {
                const recText = RECOMMENDATIONS[mat];
                let category = "Recyclable";
                if (mat === 'hazardous') category = "Hazardous";
                if (mat === 'organic') category = "Organic (Compostable)";
                
                const cleanMatName = mat.charAt(0).toUpperCase() + mat.slice(1);
                return `**${cleanMatName}** products are generally classified as **${category}**.\n\n*Recommendation*: ${recText}`;
            }
        }
        
        // 3. Check match against YOLO classes
        let matchedCls = null;
        const normalizedQuery = cleanQuery.replace(/\s+/g, '_').replace(/-/g, '_');
        
        // Sort by length desc to prevent short names like 'can' from matching in longer sentences
        const classNamesSorted = Object.keys(CLASS_TO_REC_KEY).sort((a, b) => b.length - a.length);
        for (let clsName of classNamesSorted) {
            // Word match check
            const clsWords = clsName.split('_');
            const isMatch = clsWords.every(word => {
                if (word === 'can') {
                    return new RegExp('\\bcan\\b').test(cleanQuery);
                }
                return cleanQuery.includes(word);
            });
            
            if (isMatch) {
                // Override 'can' verb false positives when other terms are present
                if (clsName === 'can' && (cleanQuery.includes('cardboard') || cleanQuery.includes('paper') || cleanQuery.includes('plastic') || cleanQuery.includes('glass') || cleanQuery.includes('box'))) {
                    continue;
                }
                matchedCls = clsName;
                break;
            }
        }
        
        if (matchedCls) {
            const recKey = CLASS_TO_REC_KEY[matchedCls];
            const recText = RECOMMENDATIONS[recKey] || "Dispose of responsibly in the general waste bin.";
            
            let category = "Non-Recyclable";
            if (RECYCLABLE.includes(matchedCls)) {
                category = "Recyclable";
            } else if (HAZARDOUS.includes(matchedCls)) {
                category = "Hazardous";
            }
            
            const cleanClsName = matchedCls.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            return `**${cleanClsName}** is classified as **${category}**.\n\n*Recommendation*: ${recText}`;
        }
        
        return "I'm not fully sure about that item. Generally, if it is clean paper, plastic bottle, metal, or glass, it is recyclable. If dirty, contaminated, or organic, dispose of it in general waste or compost.";
    }
    
    function handleSendMessage(text, query) {
        if (!text || text.trim() === '') return;
        const searchQuery = query || text;
        
        chatHistory.push({ role: 'user', text: text });
        appendMessageHTML('user', text);
        saveChatHistory();
        scrollToBottom();
        chatInput.value = '';
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing-message';
        typingDiv.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
        
        setTimeout(() => {
            const typingMsg = document.querySelector('.typing-message');
            if (typingMsg) typingMsg.remove();
            
            const reply = processAIQuery(searchQuery);
            chatHistory.push({ role: 'bot', text: reply });
            appendMessageHTML('bot', reply);
            saveChatHistory();
            scrollToBottom();
        }, 500);
    }
    
    clearBtn.addEventListener('click', () => {
        chatHistory = [
            {
                role: 'bot',
                text: '🧹 Conversation cleared! Ask me anything about recycling.'
            }
        ];
        saveChatHistory();
        renderMessages();
    });
    
    sendBtn.addEventListener('click', () => {
        handleSendMessage(chatInput.value);
    });
    
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            handleSendMessage(chatInput.value);
        }
    });
    
    window.addEventListener('DOMContentLoaded', () => {
        initIframeStyle();
        loadChatHistory();
        renderSuggestions();
    });
</script>
</body>
</html>
"""

chatbot_html = (
    chatbot_html_template
    .replace("__ECO_KNOWLEDGE_BASE__", kb_json)
    .replace("__CLASS_TO_REC_KEY__", class_rec_json)
    .replace("__RECOMMENDATIONS__", rec_json)
    .replace("__RECYCLABLE__", recyclable_json)
    .replace("__HAZARDOUS__", hazardous_json)
)

st_html(chatbot_html, height=100)