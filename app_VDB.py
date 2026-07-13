import streamlit as st
import time
import pandas as pd
import uuid
import re
from datetime import datetime
import plotly.express as px
import chromadb

# -------------------------------------------------------------------
# Smart Mock LLM Engine (Acts as a raw entity extractor)
# -------------------------------------------------------------------
def mock_llm_extract_entities(text):
    text_lower = text.lower()
    
    # 1. Dynamic Policy Number Extraction
    policy_match = re.search(r"pol-\d+", text_lower)
    policy = policy_match.group(0).upper() if policy_match else "POL-9988112"
    
    # 2. Extract RAW, uncorrected vehicle descriptions using a relaxed keyword list
    raw_vehicle = "Unknown Vehicle"
    vehicle_keywords = ["porsche", "porse", "porshe", "911", "tesla", "tesl", "tsla", "camry", "toyota", "ford", "f-150", "f150"]
    for kw in vehicle_keywords:
        if kw in text_lower:
            start_idx = max(0, text_lower.find(kw) - 7)
            end_idx = min(len(text_lower), text_lower.find(kw) + len(kw) + 12)
            raw_vehicle = text[start_idx:end_idx].strip()
            break

    # 3. Extract RAW, uncorrected broken parts phrases based on descriptive words
    raw_parts = []
    if "bumper" in text_lower or "bum" in text_lower:
        if "front" in text_lower:
            match = re.search(r"([^,\.\n]*front\s+\w+)", text_lower)
            raw_parts.append(match.group(0) if match else "front bumper")
        if "rear" in text_lower or "back" in text_lower:
            match = re.search(r"([^,\.\n]*(?:rear|backed|scratched)\s+\w+)", text_lower)
            raw_parts.append(match.group(0) if match else "rear bumper")
            
    if "light" in text_lower or "headlight" in text_lower or "tail" in text_lower:
        if "head" in text_lower or "matrix" in text_lower:
            match = re.search(r"([^,\.\n]*(?:matrix|head)\s+\w+)", text_lower)
            raw_parts.append(match.group(0) if match else "matrix headlight")
        if "tail" in text_lower or "shatter" in text_lower or "pole" in text_lower:
            match = re.search(r"([^,\.\n]*(?:tail|right|shatter)\s+\w+\s*\w*)", text_lower)
            raw_parts.append(match.group(0) if match else "tail light")

    # If the text is completely vague/low-confidence fallback
    if "something" in text_lower or "hit me" in text_lower or "not sure" in text_lower:
        raw_vehicle = "something hit me unknown vehicle"
        raw_parts = []

    # Default fallback if no specific keywords caught anything
    if raw_vehicle == "Unknown Vehicle" and not raw_parts:
        raw_vehicle = "2022 Toyota Camry"
        raw_parts = ["rear bumper", "tail light"]

    return {
        "policy_number": policy,
        "raw_extracted_vehicle": raw_vehicle,
        "damaged_parts": raw_parts
    }

# -------------------------------------------------------------------
# Agentic Workflow Components
# -------------------------------------------------------------------
class FNOLIntakeAgent:
    def process(self, email_body: str):
        claim_number = f"CLM-{uuid.uuid4().hex[:8].upper()}"
        extracted_data = mock_llm_extract_entities(email_body)
        return {"claim_number": claim_number, "raw_email_body": email_body, **extracted_data}

class DamageAssessmentAgent:
    def __init__(self):
        # Initialize free in-memory local Vector Database
        self.chroma_client = chromadb.Client()
        
        # Collection 1: Vehicle Directory
        self.vehicle_collection = self.chroma_client.get_or_create_collection(name="vehicle_directory")
        # Collection 2: Parts Catalog
        self.parts_collection = self.chroma_client.get_or_create_collection(name="parts_catalog")
        
        # Seed both vector spaces if empty
        if self.vehicle_collection.count() == 0:
            self._seed_databases()
        
    def _seed_databases(self):
        # Seed Vehicles
        vehicles = ["2024 Porsche 911 Carrera", "2022 Tesla Model 3", "2023 Ford F-150", "2022 Toyota Camry"]
        self.vehicle_collection.add(
            documents=vehicles,
            ids=[f"v_{i}" for i in range(len(vehicles))]
        )
        
        # Seed Parts Catalog
        parts_data = {
            "carbon-fiber front bumper": {"labor_hours": 8, "part_cost": 2800, "rate_per_hour": 150},
            "matrix led headlight": {"labor_hours": 3, "part_cost": 3200, "rate_per_hour": 150},
            "rear bumper": {"labor_hours": 4, "part_cost": 450, "rate_per_hour": 100},
            "tail light": {"labor_hours": 1, "part_cost": 150, "rate_per_hour": 100},
            "front bumper": {"labor_hours": 5, "part_cost": 600, "rate_per_hour": 100},
        }
        docs = list(parts_data.keys())
        self.parts_collection.add(
            documents=docs,
            metadatas=[parts_data[p] for p in docs],
            ids=[f"p_{i}" for i in range(len(docs))]
        )
        
    def process(self, structured_claim: dict):
        raw_car = structured_claim.get("raw_extracted_vehicle", "Unknown Vehicle")
        raw_parts = structured_claim.get("damaged_parts", [])
        
        vector_logs = [] 
        
        # --- 1. VECTOR MATCH THE VEHICLE BRAND/MODEL ---
        v_res = self.vehicle_collection.query(query_texts=[raw_car], n_results=1)
        if v_res and v_res['documents'] and len(v_res['documents'][0]) > 0:
            resolved_vehicle = v_res['documents'][0][0]
            v_dist = v_res['distances'][0][0] if v_res['distances'] else 0.1
            v_conf = f"{max(0, round((1 - v_dist) * 100, 1))}%"
        else:
            resolved_vehicle = "Unknown Vehicle"
            v_conf = "0%"
            
        vector_logs.append({
            "Entity Type": "🚙 Vehicle Entity",
            "User Raw Phrasing": raw_car,
            "Vector DB Resolved Match": resolved_vehicle,
            "Semantic Confidence": v_conf
        })

        # --- 2. VECTOR MATCH THE DAMAGED COMPONENTS ---
        total_estimate = 0
        total_labor = 0
        total_parts_cost = 0
        breakdown = []
        
        for part in raw_parts:
            p_res = self.parts_collection.query(query_texts=[part], n_results=1)
            
            if p_res and p_res['metadatas'] and len(p_res['metadatas'][0]) > 0:
                metrics = p_res['metadatas'][0][0]
                matched_component = p_res['documents'][0][0]
                p_dist = p_res['distances'][0][0] if p_res['distances'] else 0.1
                p_conf = f"{max(0, round((1 - p_dist) * 100, 1))}%"
            else:
                metrics = {"labor_hours": 2, "part_cost": 300, "rate_per_hour": 100}
                matched_component = part
                p_conf = "Fallback Baseline"

            labor_cost = int(metrics["labor_hours"]) * int(metrics["rate_per_hour"])
            cost = int(metrics["part_cost"]) + labor_cost
            total_estimate += cost
            total_labor += labor_cost
            total_parts_cost += int(metrics["part_cost"])
            
            breakdown.append({
                "Damaged Component": matched_component.title(),
                "Searched Phrase": part,
                "Labor Hours": int(metrics["labor_hours"]),
                "Total Labor Cost": labor_cost,
                "Replacement Part Cost": int(metrics["part_cost"]),
                "Total Component Cost": cost
            })
            
            vector_logs.append({
                "Entity Type": "🔧 Part Component",
                "User Raw Phrasing": part,
                "Vector DB Resolved Match": matched_component.title(),
                "Semantic Confidence": p_conf
            })
                
        return {
            "claim_number": structured_claim["claim_number"],
            "policy_number": structured_claim["policy_number"],
            "vehicle": resolved_vehicle,
            "date_of_accident": datetime.now().strftime("%Y-%m-%d"),
            "damage_estimate": total_estimate,
            "total_labor_cost": total_labor,
            "total_parts_cost": total_parts_cost,
            "breakdown": breakdown,
            "vector_logs": vector_logs
        }

class SettlementCalculationAgent:
    def process(self, assessment_data: dict, deductible: float):
        base_estimate = assessment_data["damage_estimate"]
        final_payout = max(0, base_estimate - deductible)
        confidence = 0.98 if base_estimate > 0 else 0.50
        reasoning = (
            f"Calculated gross repair cost of ${base_estimate} based on unified vector-space matching directory rates for "
            f"the {assessment_data['vehicle']}. Subtracted the policy deductible of ${deductible}, "
            f"yielding a net payout of ${final_payout}."
        )
        return {
            "base_estimate": base_estimate,
            "deductible_applied": deductible,
            "final_payout": final_payout,
            "reasoning": reasoning,
            "confidence_score": confidence
        }

# -------------------------------------------------------------------
# Streamlit Layout Configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="Agentic Claims Processing", page_icon="🤖", layout="wide")
st.title("🤖 Intelligent Claims Processing Platform")
st.markdown("---")

demo_templates = {
    "🟢 Auto-Approve 1: Toyota Rear-end Impact": 
        "Under policy POL-1234567, I am reporting that a delivery van bumped into my 2022 Toyota Camry from behind. The rear bumper is cracked and the left tail light is completely shattered.",
    
    "🟢 Auto-Approve 2: Minor Tesla Curb Scraping":
        "Claim filing under policy POL-4499002: I scraped the lower front bummper of my 2022 Tsla Modl 3 against a high curb while parking.",
    
    "🟢 Auto-Approve 3: Camry Parking Lot Pole Scrap":
        "Regarding policy POL-7744331, I accidentally backed into a concrete pole while leaving home, fracturing the right tail light on my Camry.",
        
    "🔴 Human Review 1: Exceeds Threshold (Porsche Typo)";
        "Urgent claim for policy POL-5544219: My 204 Porse was hit head-on, completely destroying the fiber front bum and the matrix headlight.",
        
    "🔴 Human Review 2: Low Confidence / Vague Context":
        "Under policy POL-1122334, something hit me while driving on the highway and I am not sure what happened, but there is a strange metal scraping noise.",
        
    "🔴 Human Review 3: Severe Engine/Structural Failure":
        "Filing for policy POL-6655443: A truck sideswiped my F-150. There is major body alignment damage and thick smoke is pouring out of the front engine block."
}

# -------------------------------------------------------------------
# Sidebar Design (Global Pipeline Controls)
# -------------------------------------------------------------------
st.sidebar.header("🛠️ Workflow Control Panel")
deductible_input = st.sidebar.number_
