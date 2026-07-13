import streamlit as st
import time
import pandas as pd
import uuid
import re
from datetime import datetime
import plotly.express as px

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
        return {
            "policy_number": policy,
            "vehicle": "2022 Toyota Camry",
            "date_of_accident": accident_date,
            "damage_severity": "Moderate",
            "damaged_parts": ["rear bumper", "tail light"]
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
        self.kb_vector_db = {
            "carbon-fiber front bumper": {"labor_hours": 8, "part_cost": 2800, "rate_per_hour": 150},
            "matrix led headlight": {"labor_hours": 3, "part_cost": 3200, "rate_per_hour": 150},
            "rear bumper": {"labor_hours": 4, "part_cost": 450, "rate_per_hour": 100},
            "tail light": {"labor_hours": 1, "part_cost": 150, "rate_per_hour": 100},
            "front bumper": {"labor_hours": 5, "part_cost": 600, "rate_per_hour": 100},
        }
        
    def process(self, structured_claim: dict):
        parts = structured_claim.get("damaged_parts", [])
        total_estimate = 0
        total_labor = 0
        total_parts_cost = 0
        breakdown = []
        
        for part in parts:
            part_lower = part.lower()
            metrics = self.kb_vector_db.get(part_lower, {"labor_hours": 2, "part_cost": 300, "rate_per_hour": 100})
            labor_cost = metrics["labor_hours"] * metrics["rate_per_hour"]
            cost = metrics["part_cost"] + labor_cost
            total_estimate += cost
            total_labor += labor_cost
            total_parts_cost += metrics["part_cost"]
            
            breakdown.append({
                "Damaged Component": part.title(),
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
            "total_labor_cost": total_labor,
            "total_parts_cost": total_parts_cost,
            "breakdown": breakdown
        }

class SettlementCalculationAgent:
    def process(self, assessment_data: dict, deductible: float):
        base_estimate = assessment_data["damage_estimate"]
        final_payout = max(0, base_estimate - deductible)
        confidence = 0.98 if base_estimate > 0 else 0.50
        reasoning = (
            f"Calculated gross repair cost of ${base_estimate} based on standard market rates for "
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
# Sidebar Design (Dynamic & Reactive Panels)
# -------------------------------------------------------------------
st.sidebar.header("🛠️ Workflow Control Panel")

deductible_input = st.sidebar.number_input("Policy Deductible ($)", min_value=0, max_value=5000, value=500, step=100)
approval_threshold = st.sidebar.slider("Human Escalation Threshold ($)", min_value=1000, max_value=10000, value=5000)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Active Pipeline Rules")
# Dynamic Side-Panel Information boxes syncing instantly with input parameters
st.sidebar.metric(label="🔒 Target Deductible", value=f"${deductible_input}")
st.sidebar.metric(label="🚀 Approval Max Limit", value=f"${approval_threshold}")

# -------------------------------------------------------------------
# Core Dashboard Mechanics
# -------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Incoming Claims Presentation")
    selected_template = st.selectbox("🎯 Quick-Select Demo Scenarios:", list(demo_templates.keys()))
    current_email_body = demo_templates[selected_template]
    user_email = st.text_area("Email Body Input Field:", value=current_email_body, height=160)
    execute_pipeline = st.button("Execute Pipeline", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Live Agent Execution Pipeline")
    status_box = st.container()
    
    if not execute_pipeline:
        status_box.info("ℹ️ Press **'Execute Pipeline'** to trigger multi-agent analysis sequence.")

if execute_pipeline:
    fnol_agent = FNOLIntakeAgent()
    damage_agent = DamageAssessmentAgent()
    settlement_agent = SettlementCalculationAgent()
    
    with col_right:
        # Step 1: FNOL
        p1 = st.status("🔄 [FNOL Intake] Reading narrative data...", expanded=False)
        claim_data = fnol_agent.process(user_email)
        time.sleep(0.5)
        p1.update(label=f"✅ FNOL Completed ({claim_data['claim_number']})", state="complete")
        
        # Step 2: RAG Parsing
        p2 = st.status("🔄 [Damage Assessment] Querying structural database...", expanded=False)
        assessment_data = damage_agent.process(claim_data)
        time.sleep(0.5)
        p2.update(label="✅ Damage Parameters Extracted", state="complete")
        
        # Step 3: Calculation Engine
        p3 = st.status("🔄 [Settlement Calculation] Assessing final payouts...", expanded=False)
        final_output = settlement_agent.process(assessment_data, deductible_input)
        time.sleep(0.5)
        p3.update(label="✅ Risk & Integrity Audit Finalized", state="complete")
        
        st.success("🎉 Multi-agent evaluation pipeline run completed successfully.")

    # -------------------------------------------------------------------
    # Central Content & Dashboard Real-Time Readout
    # -------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Final Settlement Payout & Integrity Verification")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Calculated Net Payout", value=f"${final_output['final_payout']}")
    m2.metric(label="Calculation Confidence Score", value=f"{final_output['confidence_score'] * 100}%")
    m3.info(f"**Calculated Deductible Applied:** ${final_output['deductible_applied']}")
    
    st.warning(f"💡 **Agent Calculation Reasoning:** {final_output['reasoning']}")
    
    # Dual Escalation Routing Rules Visual Outputs
    if final_output['final_payout'] > approval_threshold:
        st.error(f"⚠️ **Action Required**: Net payout (${final_output['final_payout']}) exceeds threshold of ${approval_threshold}. Claim routed to manual review.")
    elif final_output['confidence_score'] < 0.70:
        st.error(f"⚠️ **Action Required**: Confidence score is below acceptable limits ({final_output['confidence_score'] * 100}%). Details are too vague; routing to human review.")
    else:
        st.success("✨ **Auto-Approved**: Claim settlement approved within automated parameters.")

    # -------------------------------------------------------------------
    # Interactive Analytics & Charts Section
    # -------------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Extracted Metadata Matrix & Financial Charts")
    
    c_meta1, c_meta2, c_meta3, c_meta4 = st.columns(4)
    c_meta1.text_input("Extracted Policy Number", assessment_data["policy_number"], disabled=True)
    c_meta2.text_input("Vehicle Associated", assessment_data["vehicle"], disabled=True)
    c_meta3.text_input("Date of Accident", assessment_data["date_of_accident"], disabled=True)
    c_meta4.text_input("Damage Estimate Gross", f"${assessment_data['damage_estimate']}", disabled=True)
    
    if assessment_data["breakdown"]:
        df = pd.DataFrame(assessment_data["breakdown"])
        
        # Layout splitting standard dataframe alongside newly generated visual elements
        st.markdown("#### Cost Allocation Visualizations")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Pie Chart - Breakdown of Component Cost vs Labor Cost across entire claim
            pie_data = pd.DataFrame({
                "Cost Type": ["Replacement Part Cost", "Total Labor Cost"],
                "Total USD ($)": [assessment_data["total_parts_cost"], assessment_data["total_labor_cost"]]
            })
            fig_pie = px.pie(pie_data, values="Total USD ($)", names="Cost Type", 
                             title="Cost Distribution Ratio (Parts vs Labor)",
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with chart_col2:
            # Bar Chart - Price breakdown stacked per vehicle component
            fig_bar = px.bar(df, x="Damaged Component", y="Total Component Cost",
                             text_auto='.2s', title="Total Cost Stacked by Component",
                             labels={"Total Component Cost": "Cost ($)"},
                             color_discrete_sequence=["#1f77b4"])
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Detailed Pricing Audit Grid")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No specific damaged components recognized by the database; dynamic charting skipped.")
