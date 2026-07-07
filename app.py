import streamlit as st
import time
import json
import pandas as pd

# -------------------------------------------------------------------
# Smart Mock LLM Engine (Handles all 14 test cases dynamically)
# -------------------------------------------------------------------
def mock_llm_extract_entities(text):
    time.sleep(1.2) # Simulate processing time
    text_lower = text.lower()
    
    # --- HUMAN REVIEW & SEVERE CASES (High Valuation) ---
    if "porsche" in text_lower or "911" in text_lower:
        return {
            "vehicle_make": "Porsche", "vehicle_model": "911 Carrera",
            "damage_severity": "Severe", "impact_type": "Luxury Asset Impact",
            "damaged_parts": ["carbon-fiber front bumper", "matrix led headlight"]
        }
    elif "tesla" in text_lower and ("front" in text_lower and "rear" in text_lower):
        return {
            "vehicle_make": "Tesla", "vehicle_model": "Model 3",
            "damage_severity": "Severe", "impact_type": "Multi-car highway pileup",
            "damaged_parts": ["front bumper", "rear bumper", "windshield"]
        }
    elif "wrangler" in text_lower or "frame" in text_lower:
        return {
            "vehicle_make": "Jeep", "vehicle_model": "Wrangler",
            "damage_severity": "Critical", "impact_type": "Head-on collision",
            "damaged_parts": ["front bumper", "structural frame rails"]
        }
    elif "flood" in text_lower or "submerged" in text_lower:
        return {
            "vehicle_make": "Ford", "vehicle_model": "F-150 (Flood)",
            "damage_severity": "Critical", "impact_type": "Natural Disaster / Water Intrusion",
            "damaged_parts": ["electrical dashboard", "engine components"]
        }
    elif "civic" in text_lower or "wipeout" in text_lower:
        return {
            "vehicle_make": "Honda", "vehicle_model": "Civic",
            "damage_severity": "Severe", "impact_type": "Fixed Object Scraping",
            "damaged_parts": ["front bumper", "front door", "rear door", "rear bumper"]
        }
        
    # --- STANDARD AUTO-APPROVED CASES (Low to Moderate Valuation) ---
    elif "accord" in text_lower or "deer" in text_lower:
        return {
            "vehicle_make": "Honda", "vehicle_model": "Accord",
            "damage_severity": "Moderate", "impact_type": "Animal Strike",
            "damaged_parts": ["front bumper", "front grille"]
        }
    elif "branch" in text_lower or ("ford" in text_lower and "windshield" in text_lower and "hood" in text_lower):
        return {
            "vehicle_make": "Ford", "vehicle_model": "F-150 (Storm)",
            "damage_severity": "Moderate", "impact_type": "Weather Falling Object",
            "damaged_parts": ["windshield", "hood"]
        }
    elif "cherokee" in text_lower or "sideswipe" in text_lower:
        return {
            "vehicle_make": "Jeep", "vehicle_model": "Grand Cherokee",
            "damage_severity": "Minor", "impact_type": "Parking Lot Sideswipe",
            "damaged_parts": ["front passenger door", "side view mirror"]
        }
    elif "equinox" in text_lower or "mailbox" in text_lower:
        return {
            "vehicle_make": "Chevrolet", "vehicle_model": "Equinox",
            "damage_severity": "Minor", "impact_type": "Low-speed stationary tap",
            "damaged_parts": ["rear bumper"]
        }
    elif "subaru" in text_lower or "t-bone" in text_lower
