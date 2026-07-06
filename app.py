import streamlit as st
import time
import json
import random

# Mock LLM calls to keep the code runnable out-of-the-box.
# Replace these with actual wrapper calls (e.g., openai.chat.completions) in production.
def mock_llm_extract_entities(text):
    time.sleep(1.5) # Simulate processing
    return {
        "vehicle_make": "Toyota",
        "vehicle_model": "Camry",
        "damage_severity": "Moderate",
        "impact_type": "Rear-end collision",
        "damaged_parts": ["rear bumper", "tail light"]
    }

def mock_llm_calculate_settlement(base_estimate, deductible):
    time.sleep(1.2)
    final_payout = max(0, base_estimate - deductible)
    requires_approval = final_payout > 5000
    return final_payout, requires_approval

# -------------------------------------------------------------------
# Agent 1: FNOL Intake Agent
# -------------------------------------------------------------------
class FNOLIntakeAgent:
    def __init__(self):
        self.name = "FNOL Intake Agent"
        
    def process(self, raw_report: str):
        # In real life, send `raw_report` to LLM with a structured JSON prompt
        extracted_data = mock_llm_extract_entities(raw_report)
        return extracted_data

# -------------------------------------------------------------------
# Agent 2: Damage Assessment Agent (RAG Powered)
# -------------------------------------------------------------------
class DamageAssessmentAgent:
    def __init__(self):
        self.name = "Damage Assessment Agent"
        # Mocking a Vector DB lookup table for policy benchmarks
        self.kb_vector_db = {
            "rear bumper": {"labor_hours": 4, "part_cost": 450, "rate_per_hour": 100},
            "tail light": {"labor_hours": 1, "part_cost": 150, "rate_per_hour": 100},
            "front bumper": {"labor_hours": 5, "part_cost": 600, "rate_per_hour": 100},
            "windshield": {"labor_hours": 2, "part_cost": 350, "rate_per_hour": 120}
        }
        
    def process(self, extracted_details: dict):
        time.sleep(2) # Simulate RAG vector search
        parts = extracted_details.get("damaged_parts", [])
        total_estimate = 0
        breakdown = []
        
        for part in parts:
            part_lower = part.lower()
            if part_lower in self.kb_vector_db:
                metrics = self.kb_vector_db[part_lower]
                cost = metrics["part_cost"] + (metrics["labor_hours"] * metrics["rate_per_hour"])
                total_estimate += cost
                breakdown.append({
                    "part": part,
                    "cost_estimate": cost,
                    "details": f"Parts: ${metrics['part_cost']}, Labor: {metrics['labor_hours']} hrs"
                })
            else:
                # Default fallback if RAG doesn't find an exact match
                total_estimate += 500
                breakdown.append({"part": part, "cost_estimate": 500, "details": "Standard default estimate"})
                
        return {"total_base_estimate": total_estimate, "breakdown": breakdown}

# -------------------------------------------------------------------
# Agent 3: Settlement Calculation Agent
# -------------------------------------------------------------------
class SettlementCalculationAgent:
    def __init__(self):
        self.name = "Settlement Calculation Agent"
        
    def process(self, assessment_data: dict, deductible: float):
        base_estimate = assessment_data["total_base_estimate"]
        final_payout, requires_approval = mock_llm_calculate_settlement(base_estimate, deductible)
        
        return {
            "base_estimate": base_estimate,
            "deductible_applied": deductible,
            "final_payout": final_payout,
            "requires_human_approval": requires_approval
        }

# -------------------------------------------------------------------
# Central Manager Agent
# -------------------------------------------------------------------
class CentralManager:
    def __init__(self):
        self.fnol_agent = FNOLIntakeAgent()
        self.damage_agent = DamageAssessmentAgent()
        self.settlement_agent = SettlementCalculationAgent()
        
    def run_workflow(self, raw_report: str, deductible: float, status_container):
        # Step 1: FNOL Intake
        status_container.info("🔄 **[Manager]** Dispatching task to **FNOL Intake Agent**...")
        fnol_results = self.fnol_agent.process(raw_report)
        status_container.success("✅ **[FNOL Intake Agent Completed]** Structured data extracted successfully.")
        st.json(fnol_results)
        
        # Step 2: Damage Assessment (RAG)
        status_container.info("🔄 **[Manager]** Routing details to **Damage Assessment Agent** for RAG lookup...")
        damage_results = self.damage_agent.process(fnol_results)
        status_container.success("✅ **[Damage Assessment Agent Completed]** Standard policy benchmarks retrieved.")
        st.write(damage_results)
        
        # Step 3: Settlement Calculation
        status_container.info("🔄 **[Manager]** Submitting calculations to **Settlement Agent**...")
        settlement_results = self.settlement_agent.process(damage_results, deductible)
        status_container.success("✅ **[Settlement Calculation Agent Completed]** Final figures computed.")
        
        return settlement_results

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Agentic Claims Processing", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent Insurance Claims Processor")
st.markdown("This system uses a **Central Manager** to dynamically delegate tasks to 3 specialized agents.")

# Sidebar Configuration Inputs
st.sidebar.header("Workflow Settings")
deductible_input = st.sidebar.number_input("Policy Deductible ($)", min_value=0, max_value=5000, value=500, step=100)
approval_threshold = st.sidebar.slider("Human Approval Threshold ($)", min_value=1000, max_value=10000, value=5000)

st.subheader("1. Provide Accident Incident Report")
default_report = "I was waiting at a red light when a delivery van bumped into my 2022 Toyota Camry from behind. The rear bumper is cracked and the left tail light is completely shattered."
user_report = st.text_area("Raw Claims Narrative Text:", value=default_report, height=120)

if st.button("Run Agentic Pipeline", type="primary"):
    st.markdown("---")
    st.subheader("2. Live Agent Execution & Steps Progress")
    
    # Placeholder container for the central manager status messages
    status_box = st.empty()
    
    # Initialize Manager
    manager = CentralManager()
    
    # Execute Pipeline
    final_output = manager.run_workflow(user_report, deductible_input, status_box)
    
    # Clean final status output
    status_box.success("🎉 **[Workflow Finished]** All agents successfully finalized their sub-tasks.")
    
    # Display final UI Results Box cleanly
    st.markdown("---")
    st.subheader("3. Final Settlement Verdict")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Base Repair Estimate", value=f"${final_output['base_estimate']}")
    col2.metric(label="Deductible Applied", value=f"-${final_output['deductible_applied']}")
    col3.metric(label="Final Calculated Payout", value=f"${final_output['final_payout']}")
    
    # Conditional logic for human routing agent rules
    if final_output['final_payout'] > approval_threshold:
        st.error(f"⚠️ **Action Required**: Payout exceeds the ${approval_threshold} threshold. This claim has been automatically routed to a human manager for manual approval.")
    else:
        st.success("✨ **Auto-Approved**: Payout lies within authorized structural agent parameters. Processing payment.")
