from geopy.geocoders import Nominatim
import requests

# -----------------------------------------------------
# SMART GET USER LOCATION (GPS + Fallback)
# -----------------------------------------------------
def get_user_location():
    """
    Returns (latitude, longitude) using:
    1. Browser GPS (navigator.geolocation)
    2. Fallback → IP-based geolocation via ipinfo.io
    3. Final fallback → ABESIT Ghaziabad
    """

    try:
        from streamlit_js_eval import streamlit_js_eval
    except Exception:
        streamlit_js_eval = None

    if streamlit_js_eval is not None:
        coords = streamlit_js_eval(
            js_expressions="""
            new Promise((resolve) => {
                if (!navigator.geolocation) {
                    resolve({error: "Geolocation unavailable"});
                    return;
                }
                navigator.geolocation.getCurrentPosition(
                    (pos) => resolve({
                        latitude: pos.coords.latitude,
                        longitude: pos.coords.longitude
                    }),
                    (err) => resolve({error: err.message})
                );
            });
            """,
            key="browser_gps"
        )

        if coords and "latitude" in coords and "longitude" in coords:
            return coords["latitude"], coords["longitude"]

    # Fallback #1 → IP-based
    try:
        ipinfo = requests.get("https://ipinfo.io/json", timeout=4).json()
        if "loc" in ipinfo:
            lat, lon = map(float, ipinfo["loc"].split(","))
            return lat, lon
    except:
        pass

    # Fallback #2 → ABESIT Ghaziabad
    return 28.6080, 77.4580 


# -----------------------------------------------------
# Reverse Geocoding with Fallback
# -----------------------------------------------------
def get_address_from_coordinates(lat, lon):
    try:
        # Check if it's the ABESIT location
        if abs(lat - 28.6080) < 0.001 and abs(lon - 77.4580) < 0.001:
            return "ABESIT Campus, NH-24, Ghaziabad"
            
        geo = Nominatim(user_agent="ecoVisionAI_v2")
        loc = geo.reverse((lat, lon), language="en", timeout=5)
        if loc:
            return loc.address
    except:
        pass
        
    return f"Area: {lat:.4f}, {lon:.4f} (Ghaziabad Region)"
