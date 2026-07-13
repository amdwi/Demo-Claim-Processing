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
# Pydantic Structural Schema for Strict LLM Outputs
# -------------------------------------------------------------------
class ClaimExtractionSchema(BaseModel):
    policy_number: str = Field(description="Extracted policy ID if present, else generate an placeholder like POL-UNKNOWN")
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
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ClaimExtractionSchema,
                temperature=0.1
            ),
        )
        import json
        extracted_data = json.loads(response.text)
        return {"claim_number": claim_number, **extracted_data}

# -------------------------------------------------------------------
# Agent 2: Damage Assessment Agent (Semantic Vector DB RAG Search)
# -------------------------------------------------------------------
class DamageAssessmentAgent:
    # 🔴 Make sure this function header is shifted 4 spaces inside the class
    def __init__(self):
        # 🔴 Make sure all code lines here are shifted 8 spaces inside the class
        import chromadb
        from chromadb.utils import embedding_functions
        
        self.chroma_client = chromadb.Client()
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Safe collection creation as fixed previously
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

    # 🔴 Make sure the next method (process) is also shifted 4 spaces inside the class
    def process(self, structured_claim: dict):
        # Your remaining damage agent processing code...
        pass        
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
            "policy_number": structured_claim["policy_number"],
            "vehicle": structured_claim["vehicle"],
            "date_of_accident": structured_claim["date_of_accident"],
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
        final_payout = max(0, base_estimate - deductible)
        confidence = 0.95 if base_estimate > 0 else 0.40
        reasoning = (
            f"Calculated gross repair valuation of ${base_estimate} using real-time automated vector repository search tags. "
            f"Subtracted standard liability deductible requirement of ${deductible}, adjusting net payout allocation parameters to exactly ${final_payout}."
        )
        return {
            "base_estimate": base_estimate,
            "deductible_applied": deductible,
            "final_payout": final_payout,
            "reasoning": reasoning,
            "confidence_score": confidence
        }

# -------------------------------------------------------------------
# Streamlit Interface Layout
# -------------------------------------------------------------------
st.set_page_config(page_title="RAG Agent Claims Engine", page_icon="🤖", layout="wide")

st.title("🤖 Live RAG + LLM Multi-Agent Claims System")
st.markdown("---")

# Global Validation Check for API Key
if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
    st.error("⚠️ **Missing Configuration:** GEMINI_API_KEY environment variable is not set.")
    st.stop()

# Persistent Global Session States to store results across tab views
if "pipeline_executed" not in st.session_state:
    st.session_state.pipeline_executed = False
    st.session_state.claim_data = None
    st.session_state.assessment_data = None
    st.session_state.settlement_results = None

# Sidebar Configurations
st.sidebar.header("System Controls Configuration")
deductible_input = st.sidebar.number_input("Policy Deductible Balance ($)", min_value=0, max_value=5000, value=500, step=100)
approval_threshold = st.sidebar.slider("Automated Wire Threshold Rules ($)", min_value=1000, max_value=10000, value=4000)

# CREATE THE THREE TABS
tab1, tab2, tab3 = st.tabs(["📥 Intake Portal", "⚙️ Agent Execution Logs", "📊 Settlement Analytics"])

# ==========================================
# TAB 1: INTAKE PORTAL
# ==========================================
with tab1:
    st.subheader("System Input Channel (Incoming Mail Stream)")
    default_email = """Subject: Urgent: Claim Submission for Policy #POL-5544219
From: marcus.porsche.owner@gmail.com

Hello, my car was involved in a parking lot fender bender yesterday. A reversing SUV scraped my 2024 Porsche 911 Carrera. The carbon-fiber front bumper is visibly split, and the front matrix led headlight lens is totally shattered. Please check how much my payout will be."""

    user_email = st.text_area("Live Processing Email Text Stream Container:", value=default_email, height=180)
    
    if st.button("Fire Pipeline Execution Engine", type="primary"):
        # Create execution loaders
        with st.spinner("Processing workflows across AI agents..."):
            fnol_agent = FNOLIntakeAgent()
            damage_agent = DamageAssessmentAgent()
            settlement_agent = SettlementCalculationAgent()
            
            # Run calculations and cache inside session state
            st.session_state.claim_data = fnol_agent.process(user_email)
            st.session_state.assessment_data = damage_agent.process(st.session_state.claim_data)
            st.session_state.settlement_results = settlement_agent.process(st.session_state.assessment_data, deductible_input)
            st.session_state.pipeline_executed = True
            
        st.success("🎉 **Pipeline Execution Complete!** Navigate to the other tabs to see logs and analytics.")

# ==========================================
# TAB 2: AGENT EXECUTION LOGS
# ==========================================
with tab2:
    st.subheader("Active Pipeline Live Execution Traces")
    
    if not st.session_state.pipeline_executed:
        st.info("💡 **Awaiting Input:** Please go to the **📥 Intake Portal** and run the engine to populate execution logs.")
    else:
        st.markdown("### 🕵️‍♂️ Agent 1: FNOL Intake Trace")
        st.success(f"✅ **Structured Claim Isolated successfully** | Reference Key generated: `{st.session_state.claim_data['claim_number']}`")
        st.json(st.session_state.claim_data)
        
        st.markdown("---")
        st.markdown("### 🔍 Agent 2: RAG Vector DB Search Logs")
        st.info(f"Identified parts list for semantic analysis: `{st.session_state.assessment_data['raw_extracted_parts']}`")
        
        # Display custom matching indicators
        for item in st.session_state.assessment_data["breakdown"]:
            st.code(f"Part: {item['Damaged Component']} ➔ Database Strategy: {item['Reference Lookup Source']}", language="text")

# ==========================================
# TAB 3: SETTLEMENT ANALYTICS
# ==========================================
with tab3:
    st.subheader("System Payout Summary Dashboard")
    
    if not st.session_state.pipeline_executed:
        st.info("💡 **Awaiting Input:** Please go to the **📥 Intake Portal** and run the engine to populate financial dashboards.")
    else:
        # Destructure items for readability
        c_data = st.session_state.claim_data
        a_data = st.session_state.assessment_data
        s_data = st.session_state.settlement_results
        
        # Row 1: High-Level Client Profile
        st.markdown("### 📋 Extracted Claim Master Profile")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.text_input("Policy ID Target", a_data["policy_number"], disabled=True)
        col_b.text_input("Vehicle Extraction Mapping", a_data["vehicle"], disabled=True)
        col_c.text_input("Accident Timeline Logged", a_data["date_of_accident"], disabled=True)
        col_d.text_input("Damage Severity Classification", c_data["damage_severity"], disabled=True)
        
        st.markdown("---")
        
        # Row 2: Metrics KPI Matrix
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Calculated Net Payout", value=f"${s_data['final_payout']}")
        col2.metric(label="Agent Validation Confidence", value=f"{s_data['confidence_score'] * 100}%")
        col3.metric(label="Deductible Handled", value=f"${s_data['deductible_applied']}", delta=f"-${s_data['deductible_applied']}", delta_color="inverse")
        
        # Conditional Alerts
        if s_data['final_payout'] > approval_threshold:
            st.error(f"🛑 **Escalation Notification:** Net valuation payout of ${s_data['final_payout']} crosses processing ceiling bounds of ${approval_threshold}. Automated transfers suspended. Pushed to manual review workflows.")
        else:
            st.success("✨ **Auto-Approve Action Executed:** Processing parameter balance verification greenlit. Funding transfer issued successfully.")
            
        st.info(f"🧠 **Settlement Agent Reasoning Output Trace:** {s_data['reasoning']}")
        
        st.markdown("---")
        
        # Row 3: Itemized Financial Table Breakdown
        st.markdown("#### Vector Search Part Resolution Breakdown Matrix")
        df = pd.DataFrame(a_data["breakdown"])
        # Dropping technical log source column for cleaner display in executive tab
        display_df = df.drop(columns=["Reference Lookup Source"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
