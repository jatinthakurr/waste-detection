import json
import requests
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

FACILITY_FILE = Path("facilities.json")

# Map YOLO class name → index
CLASS_INDEX = {
    "aluminium foil": 0, "battery": 1, "aluminium blister pack": 2, "carded blister pack": 3,
    "other plastic bottle": 4, "clear plastic bottle": 5, "glass bottle": 6, "plastic bottle cap": 7,
    "metal bottle cap": 8, "broken glass": 9, "food can": 10, "aerosol": 11, "drink can": 12,
    "toilet tube": 13, "other carton": 14, "egg carton": 15, "drink carton": 16, "corrugated carton": 17,
    "meal carton": 18, "pizza box": 19, "paper cup": 20, "disposable plastic cup": 21,
    "foam cup": 22, "glass cup": 23, "other plastic cup": 24, "food waste": 25, "glass jar": 26,
    "plastic lid": 27, "metal lid": 28, "other plastic": 29, "magazine paper": 30, "tissues": 31,
    "wrapping paper": 32, "normal paper": 33, "paper bag": 34, "plastified paper bag": 35,
    "plastic film": 36, "six pack rings": 37, "garbage bag": 38, "other plastic wrapper": 39,
    "single-use carrier bag": 40, "polypropylene bag": 41, "crisp packet": 42, "spread tub": 43,
    "tupperware": 44, "disposable food container": 45, "foam food container": 46, "other plastic container": 47,
    "plastic glooves": 48, "plastic utensils": 49, "pop tab": 50, "rope & strings": 51,
    "scrap metal": 52, "shoe": 53, "squeezable tube": 54, "plastic straw": 55, "paper straw": 56,
    "styrofoam piece": 57, "unlabeled litter": 58, "cigarette": 59
}

# Mapping common names to Taco indices
ALIASES = {
    "plastic bottle": 5, "cardboard": 17, "cardboard box": 17, "can": 12, "plastic bag": 40,
    "paper": 33, "glass": 6, "metal": 52, "plastic": 29, "organic": 25, "food": 25,
    "bottle": 5, "cup": 20, "lid": 27, "straw": 55, "battery": 1, "light bulb": 54,
    "stick": 25, "cardboard bowl": 17, "plastic box": 47, "plastic cup lid": 27,
    "snack bag": 42, "scrap plastic": 29, "scrap paper": 31, "plastic bottle cap": 7,
    "aluminium foil": 0, "food can": 10, "drink can": 12, "broken glass": 9, "glass jar": 26
}

def load_facilities():
    if FACILITY_FILE.exists():
        try:
            with open(FACILITY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("facilities", [])
        except Exception as e:
            print(f"Error loading facilities: {e}")
    return []

def haversine(lat1, lon1, lat2, lon2):
    try:
        R = 6371
        dlat = radians(float(lat2) - float(lat1))
        dlon = radians(float(lon2) - float(lon1))
        a = sin(dlat/2)**2 + cos(radians(float(lat1)))*cos(radians(float(lat2)))*sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    except:
        return 999.9

def find_recycling_centers(waste_type_name, user_lat, user_lon, top_k=3):
    if not waste_type_name:
        return []
        
    # Clean input
    clean_name = str(waste_type_name).lower().strip().replace("_", " ").replace("-", " ")

    # Find index
    waste_index = None
    if clean_name in CLASS_INDEX:
        waste_index = CLASS_INDEX[clean_name]
    elif clean_name in ALIASES:
        waste_index = ALIASES[clean_name]
    else:
        # Partial match
        for k, v in CLASS_INDEX.items():
            if clean_name in k or k in clean_name:
                waste_index = v
                break
    
    if waste_index is None:
        return []

    facilities = load_facilities()
    results = []

    INDEX_CLASS = {v: k for k, v in CLASS_INDEX.items()}

    for f in facilities:
        accepts = f.get("accepts_class_indices", [])
        if waste_index not in accepts:
            continue

        lat = f.get("lat") or f.get("latitude")
        lon = f.get("lng") or f.get("lon") or f.get("longitude")

        if lat is None or lon is None:
            continue

        dist = haversine(user_lat, user_lon, lat, lon)
        
        accepted_waste_names = []
        for idx in accepts:
            if idx in INDEX_CLASS:
                name = INDEX_CLASS[idx].replace("_", " ").title()
                accepted_waste_names.append(name)

        results.append({
            "name": f.get("name", "Unknown Facility"),
            "latitude": lat,
            "longitude": lon,
            "accepted_waste": accepted_waste_names,
            "distance_km": round(dist, 2),
            "hours": f.get("hours") or f.get("opening_hours") or "N/A"
        })

    # Sort by distance
    sorted_results = sorted(results, key=lambda x: x["distance_km"])
    return sorted_results[:top_k]
