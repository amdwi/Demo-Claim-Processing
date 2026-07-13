import streamlit as st
import time
import pandas as pd
import uuid
import re
from datetime import datetime
import plotly.express as px
import chromadb

# -------------------------------------------------------------------
# Smart Mock LLM Engine (Handles enhanced test cases dynamically)
# -------------------------------------------------------------------
def mock_llm_extract_entities(text):
    text_lower = text.lower()
    
    # Dynamic Policy Number Extraction from Email Text
    policy_match = re.search(r"pol-\d+", text_lower)
    if policy_match:
        policy = policy_match.group(0).upper()
    else:
        policy = "POL-9988112" 
        
    accident_date = datetime.now().strftime("%Y-%m-%d")
    
    if "porsche" in text_lower or "911" in text_lower:
        return {
            "policy_number": policy,
            "vehicle": "2024 Porsche 911 Carrera",
            "date_of_accident": "2026-07-10",
            "damage_severity": "Severe",
            "damaged_parts": ["carbon-fiber front bumper", "matrix led headlight"]
        }
    elif "tesla" in text_lower:
        return {
            "policy_number": policy,
            "vehicle": "2022 Tesla Model 3",
            "date_of_accident": "2026-07-12",
            "damage_severity": "Severe",
            "damaged_parts": ["front bumper", "rear bumper"]
        }
    elif "engine" in text_lower or "structural" in text_lower or "smoke" in text_lower:
        return {
            "policy_number": policy,
            "vehicle": "2023 Ford F-150",
            "date_of_accident": "2026-07-11",
            "damage_severity": "Severe",
            "damaged_parts": ["front bumper", "matrix led headlight", "carbon-fiber front bumper"] 
        }
    elif "something" in text_lower or "hit me" in text_lower or "not sure" in text_lower:
        return {
            "policy_number": policy if policy_match else "POL-UNKNOWN",
            "vehicle": "Unknown Vehicle",
            "date_of_accident": accident_date,
            "damage_severity": "Unknown",
            "damaged_parts": [] 
        }
    else:
        # We pass through text hints directly to showcase the Vector DB handling raw phrasing variations
        parts = ["rear bumper", "tail light"]
        if "scratch" in text_lower:
            parts = ["scratched rear bumper skin"]
        elif "pole" in text_lower:
            parts = ["shattered right tail light assembly"]
        return {
            "policy_number": policy,
            "vehicle": "2022 Toyota Camry",
            "date_of_accident": accident_date,
            "damage_severity": "Moderate",
            "damaged_parts": parts
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
        # Initialize ephemeral/in-memory free Chroma client
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="parts_knowledge_base")
        
        if self.collection.count() == 0:
            self._seed_knowledge_base()
        
    def _seed_knowledge_base(self):
        kb_data = {
            "carbon-fiber front bumper": {"labor_hours": 8, "part_cost": 2800, "rate_per_hour": 150},
            "matrix led headlight": {"labor_hours": 3, "part_cost": 3200, "rate_per_hour": 150},
            "rear bumper": {"labor_hours": 4, "part_cost": 450, "rate_per_hour": 100},
            "tail light": {"labor_hours": 1, "part_cost": 150, "rate_per_hour": 100},
            "front bumper": {"labor_hours": 5, "part_cost": 600, "rate_per_hour": 100},
        }
        
        documents = list(kb_data.keys())
        ids = [f"part_{i}" for i in range(len(documents))]
        metadatas = [kb_data[part] for part in documents]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    def process(self, structured_claim: dict):
        parts = structured_claim.get("damaged_parts", [])
        total_estimate = 0
        total_labor = 0
        total_parts_cost = 0
        breakdown = []
        vector_logs = []  # For frontend client transparency
        
        for part in parts:
            # Semantic search across our free vector collection
            results = self.collection.query(
                query_texts=[part],
                n_results=1
            )
            
            if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
                metrics = results['metadatas'][0][0]
                matched_component = results['documents'][0][0]
                # Lower distance score means a closer semantic vector match
                distance = results['distances'][0][0] if results['distances'] else 0.0
                match_confidence = f"{max(0, round((1 - distance) * 100, 1))}%"
            else:
                metrics = {"labor_hours": 2, "part_cost": 300, "rate_per_hour": 100}
                matched_component = part
                match_confidence = "Fallback Baseline"

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
                "User Extracted Phrasing": part,
                "Vector DB Best Match": matched_component.title(),
                "Semantic Confidence": match_confidence
            })
                
        return {
            "claim_number": structured_claim["claim_number"],
            "policy_number": structured_claim["policy_number"],
            "vehicle": structured_claim["vehicle"],
            "date_of_accident": structured_claim["date_of_accident"],
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
            f"Calculated gross repair cost of ${base_estimate} based on dynamic vector-space market rates for "
            f"{assessment_data['vehicle']}. Subtracted the standard policy deductible of ${deductible}, "
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

# Pre-configured demo examples
demo_templates = {
    "Default: Standard Camry Claim": 
        "Under policy POL-1234567, I am reporting that a delivery van bumped into my 2022 Toyota Camry from behind. The rear bumper is cracked and the left tail light is completely shattered.",
    "Demo 1 (Simple/Auto-Approve): Rear-end Bumper Scratch":
        "Regarding policy POL-8822113: Someone backed into my 2022 Toyota Camry in the grocery parking lot and scratched my rear bumper.",
    "Demo 2 (Simple/Auto-Approve): Backed into Pole":
        "For policy POL-7744331, I accidentally backed into a pole at home, shattering the right tail light on my Camry.",
    "Demo 3 (Simple/Auto-Approve): Minor Tesla Front Skirt Scraping":
        "Claim filing under policy POL-4499002: I scraped the lower front bumper of my 2022 Tesla Model 3 against a high curb while parking.",
    "Demo 4 (Human Review: Exceeds $ Threshold)":
        "Urgent claim for policy POL-5544219: My 2024 Porsche 911 was hit head-on, completely destroying the carbon-fiber front bumper and the matrix led headlight.",
    "Demo 5 (Human Review: Severe Engine/Structural Damage)":
        "Filing for policy POL-6655443: A truck sideswiped my vehicle, causing severe structural frame damage and smoke is pouring out of the engine.",
    "Demo 6 (Human Review: Low Confidence / Vague Details)":
        "Under policy POL-1122334, something hit me while driving on the highway and I am not sure what happened, but there is a strange noise."
}

# -------------------------------------------------------------------
# Sidebar Design (Global Pipeline Controls)
# -------------------------------------------------------------------
st.sidebar.header("🛠️ Workflow Control Panel")

deductible_input = st.sidebar.number_input("Policy Deductible ($)", min_value=0, max_value
