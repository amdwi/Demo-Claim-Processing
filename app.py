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
    elif "subaru" in text_lower or "t-bone" in text_lower:
        return {
            "vehicle_make": "Subaru", "vehicle_model": "Outback",
            "damage_severity": "Moderate", "impact_type": "Intersection collision",
            "damaged_parts": ["front bumper", "headlights"]
        }
    elif "elantra" in text_lower or "vandalism" in text_lower:
        return {
            "vehicle_make": "Hyundai", "vehicle_model": "Elantra",
            "damage_severity": "Minor", "impact_type": "Vandalism/Break-in",
            "damaged_parts": ["driver-side front window"]
        }
    elif "tesla" in text_lower:
        return {
            "vehicle_make": "Tesla", "vehicle_model": "Model 3",
            "damage_severity": "Moderate", "impact_type": "Rear-end collision",
            "damaged_parts": ["rear bumper"]
        }
    else:
        return {
            "vehicle_make": "Toyota", "vehicle_model": "Camry",
            "damage_severity": "Moderate", "impact_type": "Rear-end collision",
            "damaged_parts": ["rear bumper", "tail light"]
        }

# -------------------------------------------------------------------
# Agent 1: FNOL Intake Agent
# -------------------------------------------------------------------
class FNOLIntakeAgent:
    def process(self, raw_report: str):
        return mock_llm_extract_entities(raw_report)

# -------------------------------------------------------------------
# Agent 2: Damage Assessment Agent (RAG Powered)
# -------------------------------------------------------------------
class DamageAssessmentAgent:
    def __init__(self):
        # Comprehensive Benchmark Knowledge Base representing Vector DB Lookups
        self.kb_vector_db = {
            # Luxury & Structural High-tier elements
            "carbon-fiber front bumper": {"labor_hours": 8, "part_cost": 2800, "rate_per_hour": 150},
            "matrix led headlight": {"labor_hours": 3, "part_cost": 3200, "rate_per_hour": 150},
            "structural frame rails": {"labor_hours": 25, "part_cost": 1800, "rate_per_hour": 120},
            "electrical dashboard": {"labor_hours": 15, "part_cost": 2500, "rate_per_hour": 110},
            "engine components": {"labor_hours": 20, "part_cost": 4500, "rate_per_hour": 110},
            
            # Standard/Mid-tier elements
            "rear bumper": {"labor_hours": 4, "part_cost": 450, "rate_per_hour": 100},
            "tail light": {"labor_hours": 1, "part_cost": 150, "rate_per_hour": 100},
            "front bumper": {"labor_hours": 5, "part_cost": 600, "rate_per_hour": 100},
            "front grille": {"labor_hours": 2, "part_cost": 250, "rate_per_hour": 100},
            "windshield": {"labor_hours": 3, "part_cost": 400, "rate_per_hour": 100},
            "hood": {"labor_hours": 4, "part_cost": 550, "rate_per_hour": 100},
            "front passenger door": {"labor_hours": 6, "part_cost": 700, "rate_per_hour": 100},
            "front door": {"labor_hours": 5, "part_cost": 600, "rate_per_hour": 100},
            "rear door": {"labor_hours": 5, "part_cost": 600, "rate_per_hour": 100},
            "side view mirror": {"labor_hours": 1, "part_cost": 200, "rate_per_hour": 100},
            "headlights": {"labor_hours": 2, "part_cost": 500, "rate_per_hour": 100},
            "driver-side front window": {"labor_hours": 2, "part_cost": 250, "rate_per_hour": 100}
        }
        
    def process(self, extracted_details: dict):
        time.sleep(1.5) 
        parts = extracted_details.get("damaged_parts", [])
        total_estimate = 0
        breakdown = []
        
        for part in parts:
            part_lower = part.lower()
            if part_lower in self.kb_vector_db:
                metrics = self.kb_vector_db[part_lower]
                labor_cost = metrics["labor_hours"] * metrics["rate_per_hour"]
                part_cost = metrics["part_cost"]
                cost = part_cost + labor_cost
                total_estimate += cost
                
                breakdown.append({
                    "Damaged Component": part.title(),
                    "Labor Hours": metrics["labor_hours"],
                    "Labor Rate ($/hr)": f"${metrics['rate_per_hour']}",
                    "Total Labor Cost": labor_cost,
                    "Replacement Part Cost": part_cost,
                    "Total Component Cost": cost
                })
            else:
                total_estimate += 500
                breakdown.append({
                    "Damaged Component": part.title(),
                    "Labor Hours": 2,
                    "Labor Rate ($/hr)": "$100",
                    "Total Labor Cost": 200,
                    "Replacement Part Cost": 300,
                    "Total Component Cost": 500
                })
                
        return {"total_base_estimate": total_estimate, "breakdown": breakdown}

# -------------------------------------------------------------------
# Agent 3: Settlement Calculation Agent
# -------------------------------------------------------------------
class SettlementCalculationAgent:
    def process(self, assessment_data: dict, deductible: float):
        time.sleep(1.0)
        base_estimate = assessment_data["total_base_estimate"]
        final_payout = max(0, base_estimate - deductible)
        return {
            "base_estimate": base_estimate,
            "deductible_applied": deductible,
            "final_payout": final_payout
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
        # Step 1: FNOL
        status_container.info("🔄 **[Manager]** Parsing unstructured narrative data via **FNOL Intake Agent**...")
        fnol_results = self.fnol_agent.process(raw_report)
        status_container.success("✅ **[FNOL Intake Agent Completed]** Operational data structures isolated.")
        st.json(fnol_results)
        
        # Step 2: Damage RAG Assessment Table
        status_container.info("🔄 **[Manager]** Querying policy vector space via **Damage Assessment Agent**...")
        damage_results = self.damage_agent.process(fnol_results)
        status_container.success("✅ **[Damage Assessment Agent Completed]** Reference standard benchmarks matched.")
        
        # Displaying data as a clean auditing table instead of JSON
        st.markdown("### 📋 RAG Database Itemized Calculation Table")
        df = pd.DataFrame(damage_results["breakdown"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Step 3: Settlement Calculations
        status_container.info("🔄 **[Manager]** Running policy rules via **Settlement Agent**...")
        settlement_results = self.settlement_agent.process(damage_results, deductible)
        status_container.success("✅ **[Settlement Calculation Agent Completed]** Account balances updated.")
        
        return settlement_results

# -------------------------------------------------------------------
# Streamlit UI Configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="Agentic Claims Processing", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent Insurance Claims Processor")
st.markdown("Centralized Manager workflow delegating tracking metrics dynamically across 3 separate software agents.")

# Sidebar Adjustments
st.sidebar.header("Workflow Control Panel")
deductible_input = st.sidebar.number_input("Policy Deductible ($)", min_value=0, max_value=5000, value=500, step=100)
approval_threshold = st.sidebar.slider("Human Escalation Threshold ($)", min_value=1000, max_value=10000, value=5000)

# Quick Copy-Paste Examples expander
with st.expander("💡 Click here to copy test inputs (Auto-Approve vs Human Review cases)"):
    st.markdown("""
    **Auto-Approve Cases (Low to Moderate Claims):**
    * `I was waiting at a red light when a delivery van bumped into my 2022 Toyota Camry from behind. The rear bumper is cracked and the left tail light is completely shattered.`
    * `Struck a deer with my 2021 Honda Accord, crushing the front bumper and cracking the front grille.`
    * `A heavy tree branch fell onto my 2023 Ford F-150, shattering the windshield and denting the hood.`
    
    **Human Review Cases (High Value / Anomalies / Structural Damage):**
    * `Got caught in a pileup in my 2022 Tesla Model 3, completely destroying both the front bumper and the rear bumper.`
    * `A shopping cart dented the carbon-fiber front bumper and smashed the matrix LED headlight on my 2024 Porsche 911.`
    * `Hit an SUV head-on in my 2021 Jeep Wrangler, crushing the front bumper and bending the main structural frame rails.`
    """)

st.subheader("1. Enter Accident Claims Text")
default_text = "I was waiting at a red light when a delivery van bumped into my 2022 Toyota Camry from behind. The rear bumper is cracked and the left tail light is completely shattered."
user_report = st.text_area("Narrative Input Field:", value=default_text, height=100)

if st.button("Execute Pipeline", type="primary"):
    st.markdown("---")
    st.subheader("2. Live Agent Execution & Steps Progress")
    
    status_box = st.empty()
    manager = CentralManager()
    final_output = manager.run_workflow(user_report, deductible_input, status_box)
    
    status_box.success("🎉 **[Workflow Finished]** High-integrity pipeline completed successfully.")
    
    # Financial Analytics Dashboard section
    st.markdown("---")
    st.subheader("3. Final Settlement Verdict")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Base Repair Estimate (Sum of Table)", value=f"${final_output['base_estimate']}")
    col2.metric(label="Deductible Deducted", value=f"-${final_output['deductible_applied']}")
    col3.metric(label="Net Calculated Payout", value=f"${final_output['final_payout']}")
    
    # Conditional Escalation Alert Box Logic
    if final_output['final_payout'] > approval_threshold:
        st.error(f"⚠️ **Action Required (Human Review Escalation)**: Net payout of ${final_output['final_payout']} exceeds your configured threshold rule of ${approval_threshold}. Auto-approval suspended. Claim flagged and routed to a human adjuster.")
    else:
        st.success("✨ **Auto-Approved**: Calculated claim settlement is inside allowed automated guidelines. Submitting wire transfer.")
