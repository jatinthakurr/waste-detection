# settings.py

# ML Model path (absolute)
DETECTION_MODEL = "C:/Users/jatin/Desktop/projects/waste-detection/weights/best.pt"

# Webcam index
WEBCAM_PATH = 0

# Waste categories
RECYCLABLE = [
    'aluminium_foil', 'aluminium_blister_pack', 'carded_blister_pack',
    'other_plastic_bottle', 'clear_plastic_bottle', 'glass_bottle',
    'plastic_bottle_cap', 'metal_bottle_cap', 'food_can', 'aerosol',
    'drink_can', 'toilet_tube', 'other_carton', 'egg_carton',
    'drink_carton', 'corrugated_carton', 'meal_carton', 'pizza_box',
    'paper_cup', 'glass_jar', 'metal_lid', 'magazine_paper',
    'wrapping_paper', 'normal_paper', 'paper_bag', 'plastified_paper_bag',
    'cardboard_box', 'can', 'plastic_bottle', 'reuseable_paper', 'scrap_metal'
]

NON_RECYCLABLE = [
    'broken_glass', 'disposable_plastic_cup', 'foam_cup', 'glass_cup',
    'other_plastic_cup', 'food_waste', 'plastic_lid', 'other_plastic',
    'tissues', 'plastic_film', 'six_pack_rings', 'garbage_bag',
    'other_plastic_wrapper', 'single-use_carrier_bag', 'polypropylene_bag',
    'crisp_packet', 'spread_tub', 'tupperware', 'disposable_food_container',
    'foam_food_container', 'other_plastic_container', 'plastic_glooves',
    'plastic_utensils', 'pop_tab', 'rope_&_strings', 'shoe',
    'plastic_straw', 'paper_straw', 'styrofoam_piece', 'unlabeled_litter',
    'cigarette', 'plastic_bag', 'scrap_paper', 'stick',
    'plastic_cup', 'snack_bag', 'plastic_box', 'straw', 'plastic_cup_lid',
    'scrap_plastic', 'cardboard_bowl', 'plastic_cultery'
]

HAZARDOUS = [
    'battery', 'squeezable_tube', 'chemical_spray_can',
    'chemical_plastic_bottle', 'chemical_plastic_gallon',
    'light_bulb', 'paint_bucket'
]

# Waste recommendations
RECOMMENDATIONS = {
    'organic': "Ideal for decomposition and composting to create nutrient-rich soil.",
    'plastic': "Do not burn! Reuse as a flower or plant pot, or recycle at specialized collection points.",
    'metal': "Clean and reuse as a pencil holder, small plant pot, or recycle with metal waste.",
    'glass': "Highly recyclable! Wash thoroughly and place in the glass recycling bin.",
    'cardboard': "Recycle with paper products or reuse for storage and DIY projects.",
    'paper': "Recycle or shred for use in composting.",
    'hazardous': "Handle with care! Dispose of only at designated hazardous waste collection centers.",
    'non_recyclable': "Dispose of responsibly in the general waste bin."
}

# Mapping specific classes to recommendation keys
CLASS_TO_REC_KEY = {
    'aluminium_foil': 'metal', 'battery': 'hazardous', 'aluminium_blister_pack': 'metal',
    'carded_blister_pack': 'plastic', 'other_plastic_bottle': 'plastic', 'clear_plastic_bottle': 'plastic',
    'glass_bottle': 'glass', 'plastic_bottle_cap': 'plastic', 'metal_bottle_cap': 'metal',
    'broken_glass': 'non_recyclable', 'food_can': 'metal', 'aerosol': 'metal',
    'drink_can': 'metal', 'toilet_tube': 'cardboard', 'other_carton': 'cardboard',
    'egg_carton': 'cardboard', 'drink_carton': 'cardboard', 'corrugated_carton': 'cardboard',
    'meal_carton': 'cardboard', 'pizza_box': 'cardboard', 'paper_cup': 'paper',
    'disposable_plastic_cup': 'plastic', 'foam_cup': 'non_recyclable', 'glass_cup': 'glass',
    'other_plastic_cup': 'plastic', 'food_waste': 'organic', 'glass_jar': 'glass',
    'plastic_lid': 'plastic', 'metal_lid': 'metal', 'other_plastic': 'plastic',
    'magazine_paper': 'paper', 'tissues': 'non_recyclable', 'wrapping_paper': 'paper',
    'normal_paper': 'paper', 'paper_bag': 'paper', 'plastified_paper_bag': 'non_recyclable',
    'plastic_film': 'plastic', 'six_pack_rings': 'plastic', 'garbage_bag': 'plastic',
    'other_plastic_wrapper': 'plastic', 'single-use_carrier_bag': 'plastic', 'polypropylene_bag': 'plastic',
    'crisp_packet': 'plastic', 'spread_tub': 'plastic', 'tupperware': 'plastic',
    'disposable_food_container': 'plastic', 'foam_food_container': 'non_recyclable',
    'other_plastic_container': 'plastic', 'plastic_glooves': 'non_recyclable',
    'plastic_utensils': 'plastic', 'pop_tab': 'metal', 'rope_&_strings': 'non_recyclable',
    'scrap_metal': 'metal', 'shoe': 'non_recyclable', 'squeezable_tube': 'hazardous',
    'plastic_straw': 'plastic', 'paper_straw': 'paper', 'styrofoam_piece': 'non_recyclable',
    'unlabeled_litter': 'non_recyclable', 'cigarette': 'non_recyclable',
    'cardboard_box': 'cardboard', 'cardboard_bowl': 'cardboard', 'can': 'metal',
    'plastic_bottle': 'plastic', 'reuseable_paper': 'paper', 'scrap_paper': 'paper',
    'stick': 'organic'
}

# Environmental impact savings per category (units in kg CO2, liters water, kWh energy saved per unit recycled)
IMPACT_FACTORS = {
    'metal': {'co2': 1.62, 'water': 24.5, 'energy': 14.0},
    'plastic': {'co2': 0.83, 'water': 12.0, 'energy': 5.8},
    'glass': {'co2': 0.32, 'water': 0.9, 'energy': 1.2},
    'cardboard': {'co2': 0.65, 'water': 19.8, 'energy': 4.1},
    'paper': {'co2': 0.55, 'water': 26.0, 'energy': 4.4},
    'organic': {'co2': 0.92, 'water': 5.0, 'energy': 0.0},  # organic composting avoids landfill methane
    'hazardous': {'co2': 0.15, 'water': 2.0, 'energy': 0.5},
    'non_recyclable': {'co2': 0.0, 'water': 0.0, 'energy': 0.0}
}

# Local knowledge base for common recycling queries
ECO_KNOWLEDGE_BASE = {
    "pizza box": "If the pizza box is greasy or has stuck cheese, the greasy parts cannot be recycled and should be composted or put in general waste. Clean cardboard parts can be recycled!",
    "glass": "Glass bottles and jars are 100% recyclable and can be recycled endlessly. Wash them first. Broken glass or mirrors should not go in the recycling bin as they damage sorting equipment.",
    "plastic": "Look for the resin identification code (1 to 7). Types 1 (PET) and 2 (HDPE) are highly recyclable. Thin films, plastic bags, and dirty wrappers usually belong in non-recyclable waste.",
    "paper": "Clean paper, newspapers, magazines, and mail are fully recyclable. Avoid recycling shredded paper in open bins, wet paper, or paper coated with plastic/wax.",
    "battery": "Batteries should NEVER go in the regular recycling or trash bin! They are hazardous and can cause fires. Dispose of them only at designated electronic waste/hazardous waste drop-off centers.",
    "light bulb": "Incandescent bulbs go to regular trash, but LED and fluorescent (CFL) bulbs contain toxic materials or metals and must be taken to hazardous waste collection centers.",
    "compost": "Organic matter like food scraps, coffee grounds, and garden trimmings can be composted. Avoid adding meat, dairy, or oil to home compost piles to prevent pests.",
    "metal": "Aluminum and steel cans are highly recyclable. Empty them and rinse off food residues. Aluminum foil is also recyclable if it is clean and rolled into a ball.",
    "e-waste": "Electronic waste contains valuable metals but also hazardous chemicals. Take old phones, laptops, and chargers to authorized e-waste collection centers."
}

