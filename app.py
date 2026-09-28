"""MedSupply Sentinel — simple Streamlit deployment (LLM via Groq API)."""
import os

import pandas as pd
import streamlit as st

import sentinel_core as core

st.set_page_config(page_title="MedSupply Sentinel", page_icon="💊", layout="wide")


def _secret(name: str) -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return os.getenv(name, "")


core.set_llm(_secret("GROQ_API_KEY") or None)


# ------------------------------------------------------------------ runtime
@st.cache_resource(show_spinner="Initialising database and workflow graph…")
def get_graph():
    core.init_db()
    return core.build_graph()


try:
    graph = get_graph()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Dataset problem: {exc}")
    st.info(
        "Put the Kaggle dataset `medsupply-sentinel-demo-data` under `./data` "
        "(it must contain `synthetic/`, `guidelines/`, `config/`), "
        "or set the `MEDSUPPLY_DATA_DIR` environment variable."
    )
    st.stop()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Recent runs")
    st.dataframe(core.recent_runs(10), hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ helpers
def review_table(s: dict) -> pd.DataFrame:
    rows = [
        ["Shortage", s["shortage_data"]["shortage_status"], s["shortage_data"]["source"]],
        ["Inventory", f"{s['inventory_data']['current_stock']} units", s["inventory_data"]["calculation"]],
        ["Coverage", f"{s['inventory_data']['coverage_days']} days", s["inventory_data"]["urgency"]],
        ["Synthetic patients", s["patient_impact"]["affected_patient_count"], s["patient_impact"]["reasoning"]],
        ["Guidelines", ", ".join(s["guideline_data"]["documents_found"]),
         " | ".join(s["guideline_data"]["constraints"])],
        ["Candidates", len(s["candidate_alternatives"]["candidate_alternatives"]),
         "Pharmacist review required = True"],
    ]
    return pd.DataFrame(rows, columns=["Evidence item", "Value", "Traceability / reasoning"]).astype(str)


def trace_table(s: dict) -> pd.DataFrame:
    rows = []
    for label, evs in [
        ("Shortage status", s["shortage_data"]["evidence"]),
        ("Inventory", s["inventory_data"]["evidence"]),
        ("Guideline", s["guideline_data"]["evidence"]),
        ("Patient impact", s["patient_impact"]["evidence"]),
        ("Candidate source", s["candidate_alternatives"]["supporting_evidence"]),
    ]:
        for ev in evs:
            rows.append([label, ev["source_name"], ev["locator"], ev["excerpt"][:300], ev["fact_or_analysis"]])
    return pd.DataFrame(rows, columns=["Conclusion", "Source", "Locator", "Evidence", "Type"])


# ------------------------------------------------------------------ main
st.title("💊 MedSupply Sentinel")
st.caption("Multi-agent drug-shortage decision support — synthetic demo data only.")
st.warning(
    "Candidate alternatives are for qualified pharmacist review only. This app does not prescribe, "
    "switch medication, or make autonomous clinical decisions. No approval → no purchase order."
)

meds = core.list_medications()
labels = {f"{r.drug_name} ({r.drug_id})": r.drug_id for r in meds.itertuples()}
choice = st.selectbox("Medication", list(labels))

if st.button("▶ Start shortage workflow", type="primary"):
    state, config = core.new_workflow_state(labels[choice])
    with st.spinner("Running agents (shortage → inventory + guidelines → patient impact → substitution → validator)…"):
        try:
            paused = graph.invoke(state, config=config)
        except Exception as exc:
            st.error(f"Workflow failed: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state["wf"] = {"id": state["workflow_id"], "config": config, "paused": paused, "final": None}

wf = st.session_state.get("wf")
if not wf:
    st.info("Choose a medication and start the workflow.")
    st.stop()

st.divider()
st.subheader(f"Workflow `{wf['id']}`")

# ---- Stage 1: waiting for the human gate
if wf["final"] is None:
    paused = wf["paused"]
    if not paused.get("__interrupt__"):
        st.error("The workflow did not pause at the pharmacist gate. Check the agent logs.")
        st.stop()

    st.error("🛑 PHARMACIST REVIEW REQUIRED")
    st.dataframe(review_table(paused), hide_index=True, use_container_width=True)

    with st.form("decision"):
        status = st.radio("Decision", ["APPROVED", "REJECTED", "NEEDS_MORE_INFORMATION"], horizontal=True)
        role = st.text_input("Reviewer role", value="PHARMACIST")
        reason = st.text_area("Decision reason (required)")
        submitted = st.form_submit_button("Submit decision & resume workflow")

    if submitted:
        if not reason.strip():
            st.error("Please write a decision reason.")
            st.stop()
        decision = {"status": status, "reviewer_role": role.strip() or "PHARMACIST", "reason": reason.strip()}
        with st.spinner("Resuming workflow…"):
            try:
                final = graph.invoke(core.Command(resume=decision), config=wf["config"])
                core.finalize_run(wf["id"], final)
                core.save_report(wf["id"], final["final_report"])
            except Exception as exc:
                st.error(f"Resume failed: {type(exc).__name__}: {exc}")
                st.stop()
        wf["final"] = final
        st.rerun()
    st.stop()

# ---- Stage 2: finished
final = wf["final"]
approval = final.get("approval_status")
(st.success if approval == "APPROVED" else st.warning)(f"Human decision: **{approval}**")

tab_report, tab_po, tab_notif, tab_trace, tab_logs = st.tabs(
    ["Incident report", "Purchase order", "Notifications", "Evidence trace", "Agent logs"])

with tab_report:
    st.markdown(final["final_report"])
    st.download_button("Download report (.md)", final["final_report"], file_name=f"{wf['id']}_incident_report.md")

with tab_po:
    po = final.get("purchase_order")
    if po:
        st.json(po)
        pdf_path = core.make_po_pdf(po)
        st.download_button("Download PO draft (.pdf)", pdf_path.read_bytes(),
                           file_name=pdf_path.name, mime="application/pdf")
    else:
        st.info("No purchase order — it is only generated after explicit pharmacist approval.")

with tab_notif:
    notifs = final.get("notifications", [])
    if notifs:
        for n in notifs:
            st.markdown(f"**{n['subject']}** → {n['recipient']} ({n['status']})")
            st.code(n["body"])
    else:
        st.info("No notifications were sent.")

with tab_trace:
    st.dataframe(trace_table(final), hide_index=True, use_container_width=True)

with tab_logs:
    logs = pd.DataFrame(final.get("agent_logs", []))
    if not logs.empty:
        st.dataframe(logs[["agent", "status", "timestamp", "duration_seconds", "retry_count"]],
                     hide_index=True, use_container_width=True)
