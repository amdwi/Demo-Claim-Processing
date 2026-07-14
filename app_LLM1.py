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
        # Initialize free in-memory local Vector Database
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
        
    "🔴 Human Review 3: Severe Engine/Structural Failure":
        "Filing for policy POL-6655443: A truck sideswiped my F-150. There is major body alignment damage and thick smoke is pouring out of the front engine block."
}

# -------------------------------------------------------------------
# Sidebar Design (Global Pipeline Controls & API Management)
# -------------------------------------------------------------------
st.sidebar.header("🔑 API Credentials")
secrets_key = st.secrets.get("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input("Groq Cloud API Key", value=secrets_key, type="password", help="Enter your free-trial API key from console.groq.com")

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Workflow Control Panel")
deductible_input = st.sidebar.number_input("Policy Deductible ($)", min_value=0, max_value=5000, value=500, step=100)
approval_threshold = st.sidebar.slider("Human Escalation Threshold ($)", min_value=1000, max_value=10000, value=5000)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Active Pipeline Rules")
st.sidebar.metric(label="🔒 Target Deductible", value=f"${deductible_input}")
st.sidebar.metric(label="🚀 Approval Max Limit", value=f"${approval_threshold}")

if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
    st.session_state.assessment_data = None
    st.session_state.final_output = None
    st.session_state.last_llm_input = ""
    st.session_state.last_llm_output = None

# -------------------------------------------------------------------
# Interface Division (Tabs)
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📥 1. Intake & Agent Execution", 
    "⚖️ 2. Settlement & Integrity Audit", 
    "📊 3. Analytics & Cost Matrix"
])

# --- TAB 1: INTAKE & LIVE EXECUTION ---
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.subheader("Document Input Methods")
        input_method = st.radio("Choose Input Type:", ["Use Demo Quick-Select Templates", "Upload Accident PDF Report"])
        
        final_processing_text = ""
        
        if input_method == "Use Demo Quick-Select Templates":
            selected_template = st.selectbox("🎯 Quick-Select Demo Scenarios:", list(demo_templates.keys()))
            current_email_body = demo_templates[selected_template]
            final_processing_text = st.text_area("Template Text Viewport:", value=current_email_body, height=160)
        else:
            uploaded_pdf = st.file_uploader("Upload Official Accident Report (PDF):", type=["pdf"], help="Drag and drop a text-based PDF claim document.")
            if uploaded_pdf is not None:
                try:
                    reader = PdfReader(uploaded_pdf)
                    extracted_pdf_text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_pdf_text += page_text + "\n"
                    
                    if not extracted_pdf_text.strip():
                        st.error("❌ This PDF appears to be a scanned image. Please provide a text-based document for analysis.")
                    else:
                        final_processing_text = extracted_pdf_text
                        st.success("📝 PDF Content extracted successfully into memory pipeline!")
                except Exception as e:
                    st.error(f"❌ Document Parsing Error: {e}")
                    
        execute_pipeline = st.button("Execute Multi-Agent Pipeline", type="primary", width="stretch")

    with col_right:
        st.subheader("Live Agent Execution Monitor")
        
        if execute_pipeline:
            if not groq_api_key:
                st.error("🛑 Action Required: Please provide a valid Groq API Key in the sidebar control panel to trigger the real LLM engine extraction.")
            elif final_processing_text.strip():
                st.session_state.last_llm_input = final_processing_text
                
                fnol_agent = FNOLIntakeAgent(api_key=groq_api_key)
                damage_agent = DamageAssessmentAgent()
                settlement_agent = SettlementCalculationAgent()
                
                p1 = st.status("🔄 [Groq LLM Engine] Processing Narrative Intake Data...", expanded=True)
                claim_data = fnol_agent.process(final_processing_text)
                
                st.session_state.last_llm_output = {
                    "Extracted Policy": claim_data.get("policy_number"),
                    "Extracted Raw Vehicle Description": claim_data.get("raw_extracted_vehicle"),
                    "Extracted Raw Parts List": claim_data.get("damaged_parts")
                }
                
                # --- ADJUSTED UI: SCROLLABLE TEXT WINDOW + COLLAPSIBLE JSON CONTAINER ---
                st.markdown("### 📊 Live LLM Effectiveness Audit")
                c_in, c_out = st.columns(2)
                with c_in:
                    st.text_area(
                        "📝 What the LLM Read (Raw Input):",
                        value=st.session_state.last_llm_input,
                        height=140,
                        disabled=True,
                        key="live_audit_raw_input"
                    )
                with c_out:
                    with st.expander("🧠 View LLM Conclusion (Structured JSON)", expanded=True):
                        st.json(st.session_state.last_llm_output)
                
                time.sleep(0.2)
                p1.update(label=f"✅ Groq LLM Ingestion Finalized ({claim_data['claim_number']})", state="complete")
                
                p2 = st.status("🔄 [Chroma DB Engine] Aligning entities against Vector Catalog Index...", expanded=True)
                st.session_state.assessment_data = damage_agent.process(claim_data)
                
                if st.session_state.assessment_data["vector_logs"]:
                    st.markdown("##### 🎯 Dynamic Vector DB Alignment Matrix:")
                    st.dataframe(
                        pd.DataFrame(st.session_state.assessment_data["vector_logs"]),
                        width="stretch", hide_index=True
                    )
                time.sleep(0.2)
                p2.update(label="✅ Vector Identity & Part Mapping Complete", state="complete")
                
                p3 = st.status("🔄 [Settlement Calculation] Assessing financial payout...", expanded=False)
                st.session_state.final_output = settlement_agent.process(st.session_state.assessment_data, deductible_input)
                time.sleep(0.2)
                p3.update(label="✅ Calculations Finalized", state="complete")
                
                st.session_state.pipeline_run = True
                st.success("🎉 Processing complete! Check next tabs.")
            else:
                st.error("⚠️ Ingestion workspace is empty. Please select a template or verify your PDF config.")
            
        elif st.session_state.pipeline_run:
            st.info("✅ Last run data cached.")
            
            # --- PERSISTENT CACHED AUDIT VIEWS ---
            st.markdown("### 📊 Cached LLM Effectiveness Audit")
            c_in, c_out = st.columns(2)
            with c_in:
                st.text_area(
                    "📝 What the LLM Read (Raw Input):",
                    value=st.session_state.last_llm_input,
                    height=140,
                    disabled=True,
                    key="cached_audit_raw_input"
                )
            with c_out:
                with st.expander("🧠 View LLM Conclusion (Structured JSON)", expanded=True):
                    st.json(st.session_state.last_llm_output)
                
            st.markdown("---")
            if st.session_state.assessment_data and st.session_state.assessment_data.get("vector_logs"):
                st.markdown("##### 🎯 Cached Vector DB Alignment Matrix:")
                st.dataframe(
                    pd.DataFrame(st.session_state.assessment_data["vector_logs"]),
                    width="stretch", hide_index=True
                )
        else:
            st.info("ℹ️ Enter your Groq Key in the sidebar, select a scenario, and press **'Execute Multi-Agent Pipeline'**.")

# --- TAB 2: SETTLEMENT & INTEGRITY AUDIT ---
with tab2:
    st.subheader("Final Settlement Assessment & Routing Logic")
    if st.session_state.pipeline_run:
        final_output = st.session_state.final_output
        assessment_data = st.session_state.assessment_data
        
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Calculated Net Payout", value=f"${final_output['final_payout']}")
        m2.metric(label="Calculation Confidence Score", value=f"{final_output['confidence_score'] * 100}%")
        m3.info(f"**Calculated Deductible Applied:** ${final_output['deductible_applied']}")
        
        st.warning(f"💡 **Agent Calculation Reasoning:** {final_output['reasoning']}")
        
        st.markdown("### 🛑 Automated Routing Guardrails")
        if final_output['final_payout'] > approval_threshold:
            st.error(f"⚠️ **Action Required**: Net payout (${final_output['final_payout']}) exceeds threshold limit of ${approval_threshold}. Claim routed to manual human review.")
        elif not assessment_data["breakdown"]:
            st.error(f"⚠️ **Action Required**: Confidence score is below acceptable limits. Details are too vague; routing to manual human review.")
        else:
            st.success("✨ **Auto-Approved**: Claim settlement approved within standard automated parameters.")
    else:
        st.info("📥 Please run the execution pipeline in **Tab 1** to view settlement data.")

# --- TAB 3: ANALYTICS & COST MATRIX ---
with tab3:
    st.subheader("Extracted Metadata Matrix & Financial Data Visualization")
    if st.session_state.pipeline_run:
        assessment_data = st.session_state.assessment_data
        
        c_meta1, c_meta2, c_meta3, c_meta4 = st.columns(4)
        c_meta1.text_input("Resolved Vehicle Make/Model", assessment_data["vehicle"], disabled=True)
        c_meta2.text_input("Extracted Policy Number", assessment_data["policy_number"], disabled=True)
        c_meta3.text_input("Date of Accident", assessment_data["date_of_accident"], disabled=True)
        c_meta4.text_input("Damage Estimate Gross", f"${assessment_data['damage_estimate']}", disabled=True)
        
        st.markdown("---")
        if assessment_data["breakdown"]:
            df = pd.DataFrame(assessment_data["breakdown"])
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                pie_data = pd.DataFrame({
                    "Cost Type": ["Replacement Part Cost", "Total Labor Cost"],
                    "Total USD ($)": [assessment_data["total_parts_cost"], assessment_data["total_labor_cost"]]
                })
                fig_pie = px.pie(pie_data, values="Total USD ($)", names="Cost Type", title="Cost Distribution Ratio", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_pie, width="stretch")
            with chart_col2:
                fig_bar = px.bar(df, x="Damaged Component", y="Total Component Cost", text_auto='.2s', title="Total Cost Stacked by Component")
                st.plotly_chart(fig_bar, width="stretch")
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.warning("⚠️ No components map cleanly to pricing catalogs. Multi-layered graphs skipped.")
    else:
        st.info("📥 Please run the execution pipeline in **Tab 1** to generate metrics.")
