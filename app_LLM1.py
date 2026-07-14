import streamlit as st
import time
import pandas as pd
import uuid
import re
import json
from datetime import datetime
import plotly.express as px
import chromadb
from pypdf import PdfReader
from groq import Groq 

# -------------------------------------------------------------------
# Production-Grade LLM Extraction Engine via Groq
# -------------------------------------------------------------------
def extract_entities_via_groq(text, api_key):
    """
    Leverages a live LLM to handle messy human natural language narratives,
    natively dealing with typos and complex structuring without hardcoded keywords.
    """
    if not api_key:
        return {
            "policy_number": "POL-ERR-NOKEY",
            "raw_extracted_vehicle": "Missing Groq API Key",
            "damaged_parts": []
        }
        
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        You are an advanced enterprise insurance intake AI system. Analyze the unstructured claims document or email text below.
        
        Tasks:
        1. Extract the Policy Number (usually matches patterns like POL-xxxxxxx). If none exists, output "POL-UNKNOWN".
        2. Identify the vehicle make, model, and year being reported. Extract exactly how the user wrote it (including typos).
        3. Identify a clean list of all specific damaged car parts or broken components described by the claimant. Do not standardize them yet; keep their raw descriptive variations intact.

        CRITICAL: You must respond ONLY with a valid, clean JSON object. Do not include any introductory markdown, formatting ticks like ```json, or conversational prose. 

        The JSON must follow this precise structure:
        {{
            "policy_number": "extracted string or POL-UNKNOWN",
            "raw_extracted_vehicle": "extracted raw vehicle string",
            "damaged_parts": ["raw part phrase 1", "raw part phrase 2"]
        }}

        Claims Narrative Text to Parse:
        \"\"\"{text}\"\"\"
        """
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated production supported model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,         
            response_format={"type": "json_object"}
        )
        
        structured_response = json.loads(completion.choices[0].message.content)
        return structured_response
        
    except Exception as e:
        st.error(f"Groq API Execution Error: {e}")
        return {
            "policy_number": "POL-ERROR",
            "raw_extracted_vehicle": "Extraction Timeout/Failure",
            "damaged_parts": []
        }

# -------------------------------------------------------------------
# Agentic Workflow Components
# -------------------------------------------------------------------
class FNOLIntakeAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        
    def process(self, raw_text: str):
        claim_number = f"CLM-{uuid.uuid4().hex[:8].upper()}"
        extracted_data = extract_entities_via_groq(raw_text, self.api_key)
        return {"claim_number": claim_number, "raw_text_body": raw_text, **extracted_data}

class DamageAssessmentAgent:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.vehicle_collection = self.chroma_client.get_or_create_collection(name="vehicle_directory")
        self.parts_collection = self.chroma_client.get_or_create_collection(name="parts_catalog")
        
        if self.vehicle_collection.count() == 0:
            self._seed_databases()
        
    def _seed_databases(self):
        vehicles = ["2024 Porsche 911 Carrera", "2022 Tesla Model 3", "2023 Ford F-150", "2022 Toyota Camry"]
        self.vehicle_collection.add(
            documents=vehicles,
            ids=[f"v_{i}" for i in range(len(vehicles))]
        )
        
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
st.title("🤖 Enterprise Agentic Claims Processing Platform")
st.markdown("---")

demo_templates = {
    "🟢 Auto-Approve 1: Toyota Rear-end Impact": 
        "Under policy POL-1234567, I am reporting that a delivery van bumped into my 2022 Toyota Camry from behind. The rear bumper is cracked and the left tail light is completely shattered.",
    
    "🟢 Auto-Approve 2: Minor Tesla Curb Scraping":
        "Claim filing under policy POL-4499002: I scraped the lower front bummper of my 2022 Tsla Modl 3 against a high curb while parking.",
    
    "🟢 Auto-Approve 3: Camry Parking Lot Pole Scrap":
        "Regarding policy POL-7744331, I accidentally backed into a concrete pole while leaving home, fracturing the right tail light on my Camry.",
        
    "🔴 Human Review 1: Exceeds Threshold (Porsche Typo)":
        "Urgent claim for policy POL-5544219: My 204 Porse was hit head-on, completely destroying the fiber front bum and the matrix headlight.",
        
    "🔴 Human Review 2: Low Confidence / Vague Context":
        "Under policy POL-1122334, something hit me while driving on the highway and I am not sure what happened, but there is a strange metal scraping noise.",
