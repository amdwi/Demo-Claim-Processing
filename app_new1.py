import streamlit as st
import pandas as pd
import uuid
import os
from datetime import datetime
from pydantic import BaseModel, Field

# Vector Database & LLM SDK imports
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

# -------------------------------------------------------------------
# Extraction Schema Configuration
# -------------------------------------------------------------------
class ClaimExtractionSchema(BaseModel):
    policy_number: str = Field(description="Extracted policy ID if present, else generate a placeholder like POL-UNKNOWN")
    vehicle: str = Field(description="Make, model, and year of the car")
    date_of_accident: str = Field(description="Date format YYYY-MM-DD. Use current date if not specified.")
    damage_severity: str = Field(description="Minor, Moderate, Severe, or Critical")
    damaged_parts: list[str] = Field(description="Clean, explicit array of damaged automotive components mentioned.")

# -------------------------------------------------------------------
# Agent 1: FNOL Intake Agent (Live LLM Engine)
# -------------------------------------------------------------------
class FNOLIntakeAgent:
    def __init__(self):
        self.client = genai.Client()

    def process(self, email_body: str):
        claim_number = f"CLM-{uuid.uuid4().hex[:8].upper()}"
        prompt = f"""
        You are an expert insurance adjuster intake bot. Analyze the incoming customer email narrative below. 
        Extract key operational entities precisely.
        
        Email Context:
        {email_body}
        """
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClaimExtractionSchema, # Enforces strict schema compilation
                    temperature=0.1
                ),
            )
            
            # Use SDK native parser instead of manual json.loads to bypass parsing exceptions
            extracted_data = response.parsed.model_dump()
            
        except Exception as e:
            # Fallback strategy to protect pipeline execution state against API Client errors
            st.sidebar.error(f"⚠️ GenAI Client Error: {str(e)}. Utilizing fallback extraction parameters.")
            extracted_data = {
                "policy_number": "POL-5544219",
                "vehicle": "2024 Porsche 911 Carrera",
                "date_of_accident": datetime.now().strftime("%Y-%m-%d"),
                "damage_severity": "Severe",
                "damaged_parts": ["carbon-fiber front bumper", "matrix led headlight"]
            }
            
        return {"claim_number": claim_number, **extracted_data}

# -------------------------------------------------------------------
# Agent 2: Damage Assessment Agent (Semantic Vector DB RAG Search)
# -------------------------------------------------------------------
class DamageAssessmentAgent:
    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions
        
        self.chroma_client = chromadb.Client()
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Safe collection creation
        self.collection = self.chroma_client.get_or_create_collection(
            name="part_costs_catalog",
            embedding_function=self.emb_fn
        )
        
        # If your database is empty, seed the initial catalog rules
        if self.collection.count() == 0:
            self.collection.add(
                documents=["carbon-fiber front bumper", "matrix led headlight", "rear bumper", "tail light", "front bumper"],
                ids=["part_1", "part_2", "part_3", "part_4", "part_5"],
                metadatas=[
                    {"labor_hours": 8, "part_cost": 2800, "rate_per_hour": 150},
                    {"labor_hours": 3, "part_cost": 3200, "rate_per_hour": 150},
                    {"labor_hours": 4, "part_cost": 450, "rate_per_hour": 100},
                    {"labor_hours": 1, "part_cost": 150, "rate_per_hour": 100},
                    {"labor_hours": 5, "part_cost": 600, "rate_per_hour": 100}
                ]
            )

    def process(self, structured_claim: dict):
        parts = structured_claim.get("damaged_parts", [])
        total_estimate = 0
        breakdown = []
        
        for part in parts:
            results = self.collection.query(query_texts=[part], n_results=1)
            if results['metadatas'] and len(results['metadatas'][0]) > 0:
                metrics = results['metadatas'][0][0]
                source_note = f"Semantic Match ({results['documents'][0][0]})"
            else:
                metrics = {"labor_hours": 3, "part_cost": 350, "rate_per_hour": 100}
                source_note = "Fallback Manual Default Standard"
                
            labor_cost = metrics["labor_hours"] * metrics["rate_per_hour"]
            cost = metrics["part_cost"] + labor_cost
            total_estimate += cost
            
            breakdown.append({
                "Damaged Component": part.title(),
                "Reference Lookup Source": source_note,
                "Labor Hours": metrics["labor_hours"],
                "Total Labor Cost": labor_cost,
                "Replacement Part Cost": metrics["part_cost"],
                "Total Component Cost": cost
            })
                
        return {
            "claim_number": structured_claim["claim_number"],
            "policy_number": structured_claim.get("policy_number", "POL-UNKNOWN"),
            "vehicle": structured_claim.get("vehicle", "Unknown Vehicle"),
            "date_of_accident": structured_claim.get("date_of_accident", datetime.now().strftime("%Y-%m-%d")),
            "damage_estimate": total_estimate,
            "breakdown": breakdown,
            "raw_extracted_parts": parts
        }

# -------------------------------------------------------------------
# Agent 3: Settlement Calculation Agent (Mathematical Audit & Analysis)
# -------------------------------------------------------------------
class SettlementCalculationAgent:
    def process(self, assessment_data: dict, deductible: float):
        base_estimate = assessment_data["damage_estimate"]
        final_payout = max(
