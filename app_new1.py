import streamlit as st
import time
import pandas as pd
import uuid
from datetime import datetime

# -------------------------------------------------------------------
# Smart Mock LLM Engine (Handles enhanced test cases dynamically)
# -------------------------------------------------------------------
def mock_llm_extract_entities(text):
    time.sleep(1.2) 
    text_lower = text.lower()
    
    # Defaults
    policy = "POL-9988112"
    accident_date = datetime.now().strftime("%Y-%m-%d")
    
    if "porsche" in text_lower or "911" in text_lower:
        return {
            "policy_number": "POL-5544219",
            "vehicle": "2024 Porsche 911 Carrera",
            "date_of_accident": "2026-07-10",
            "damage_severity": "Severe",
            "damaged_parts": ["carbon-fiber front bumper", "matrix led headlight"]
        }
    elif "tesla" in text_lower:
        return {
            "policy_number": "POL-3322110",
            "vehicle": "2022 Tesla Model 3",
            "date_of_accident": "2026-07-12",
            "damage_severity": "Severe",
            "damaged_parts": ["front bumper", "rear bumper"]
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
# Agent 1: FNOL Intake Agent (Email Input & Claim Gen)
# -------------------------------------------------------------------
class FNOLIntakeAgent:
    def process(self, email_body: str):
        # Generate unique claim number
        claim_number = f"CLM-{uuid.uuid4().hex[:8].upper()}"
        
        # Extract metadata via mock LLM
        extracted_data = mock_llm_extract_entities(email_body)
        
        # Structure the final claim output
        structured_claim = {
            "claim_number": claim_number,
            "raw_email_body": email_body,
            **extracted_data
        }
        return structured_claim

# -------------------------------------------------------------------
# Agent 2: Damage Assessment Agent (RAG Powered)
# -------------------------------------------------------------------
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
        time.sleep(1.5) 
        parts = structured_claim.get("damaged_parts", [])
        total_estimate = 0
        breakdown = []
        
        for part in parts:
            part_lower = part.lower()
            metrics = self.kb_vector_db.get(part_lower, {"labor_hours": 2, "part_cost": 300, "rate_per_hour": 100})
            labor_cost = metrics["labor_hours"] * metrics["rate_per_hour"]
            cost = metrics["part_cost"] + labor_cost
            total_estimate += cost
            
            breakdown.append({
                "Damaged Component": part.title(),
                "Labor Hours": metrics["labor_hours"],
                "Total Labor Cost": labor_cost,
                "Replacement Part Cost": metrics["part_cost"],
                "Total Component Cost": cost
            })
                
        # Return enhanced profile requested for Agent 2
        return {
            "claim_number": structured_claim["claim_number"],
            "policy_number": structured_claim["policy_number"],
            "vehicle": structured_claim["vehicle"],
            "date_of_accident": structured_claim["date_of_accident"],
            "damage_estimate": total_estimate,
            "breakdown": breakdown
        }

# -------------------------------------------------------------------
# Agent 3: Settlement Calculation Agent (Reasoning & Confidence)
# -------------------------------------------------------------------
class SettlementCalculationAgent:
    def process(self, assessment_data: dict, deductible: float):
        time.sleep(1.0)
        base_estimate = assessment_data["damage_estimate"]
        final_payout = max(0, base_estimate - deductible)
        
        # Dynamically evaluate confidence based on missing parts in DB
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
# Central Manager Agent
# -------------------------------------------------------------------
class CentralManager:
    def __init__(self):
        self.fnol_agent = FNOLIntakeAgent()
        self.damage_agent = DamageAssessmentAgent()
        self.settlement_agent = SettlementCalculationAgent()
        
    def run_workflow(self, email_input: str, deductible: float, status_container):
        # Step 1: FNOL Intake
        status_container.info("🔄 **[Manager]** Parsing incoming email narrative via **FNOL Intake Agent**...")
        claim_data = self.fnol_agent.process(email_input)
        status_container.success(f"✅ **[FNOL Intake Completed]** Claim Number {claim_data['claim_number']} Generated.")
        
        # Step 2: Damage Assessment
        status_container.info("🔄 **[Manager]** Pulling parameters & pricing benchmarks via **Damage Agent**...")
        assessment_data = self.damage_agent.process(claim_data)
        
        st.markdown("### 📋 Agent 2 Output Matrix")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.text_input("Policy Number", assessment_data["policy_number"], disabled=True)
        col_b.text_input("Vehicle Associated", assessment_data["vehicle"], disabled=True)
        col_c.text_input("Date of Accident", assessment_data["date_of_accident"], disabled=True)
        col_d.text_input("Damage Estimate Gross", f"${assessment_data['damage_estimate']}", disabled=True)
        
        df = pd.DataFrame(assessment_data["breakdown"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Step 3: Settlement Calculations
        status_container.info("🔄 **[Manager]** Assessing final payout limits via **Settlement Agent**...")
        settlement_results = self.settlement_agent.process(assessment_data, deductible)
        status_container.success("✅ **[Settlement Calculation Agent Completed]** Risk scoring derived.")
        
        return settlement_results

# -------------------------------------------------------------------
# Streamlit UI Configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="Agentic Claims Processing", page_icon="🤖", layout="wide")
st.title("🤖 Intelligent Claims Processing Platform")

st.sidebar.header("Workflow Control Panel")
deductible_input = st.sidebar.number_input("Policy Deductible ($)", min_value=0, max_value=5000, value=500, step=100)
approval_threshold = st.sidebar.slider("Human Escalation Threshold ($)", min_value=1000, max_value=10000, value=5000)

st.subheader("1. Incoming Claims Email Processing")
default_email = """Subject: Accident Claim Submission
From: driver@email.com

Hello, I am writing to report an incident. I was waiting at a red light when a delivery van bumped into my 2022 Toyota Camry from behind. The rear bumper is cracked and the left tail light is completely shattered."""

user_email = st.text_area("Email Body Input Field:", value=default_email, height=140)

if st.button("Execute Pipeline", type="primary"):
    st.markdown("---")
    st.subheader("2. Live Agent Execution & Steps Progress")
    
    status_box = st.empty()
    manager = CentralManager()
    final_output = manager.run_workflow(user_email, deductible_input, status_box)
    
    status_box.success("🎉 **[Workflow Finished]** Multi-agent evaluation completed.")
    
    # Financial Analytics Dashboard section
    st.markdown("---")
    st.subheader("3. Final Settlement Payout & Integrity Verification")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Calculated Net Payout", value=f"${final_output['final_payout']}")
    col2.metric(label="Calculation Confidence Score", value=f"{final_output['confidence_score'] * 100}%")
    col3.info(f"**Calculated Deductible Applied:** ${final_output['deductible_applied']}")
    
    st.warning(f"💡 **Agent Calculation Reasoning:** {final_output['reasoning']}")
    
    if final_output['final_payout'] > approval_threshold:
        st.error(f"⚠️ **Action Required**: Net payout exceeds threshold of ${approval_threshold}. Claim routed to manual review.")
    else:
        st.success("✨ **Auto-Approved**: Claim settlement approved within automated parameters.")
