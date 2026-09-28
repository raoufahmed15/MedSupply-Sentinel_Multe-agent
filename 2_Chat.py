"""💬 Chat — MedSupply Assistant (notebook 17B + 17C, tab 2)."""
import os

import streamlit as st

import chat_engine as ce
import sentinel_core as core

st.set_page_config(page_title="Chat · MedSupply Sentinel", page_icon="💬", layout="wide")


def _secret(name: str) -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return os.getenv(name, "")


api_key = _secret("GROQ_API_KEY")
llm = None
if api_key:
    try:
        llm = ce.GroqChat(api_key=api_key)
    except Exception as exc:
        st.sidebar.error(f"Groq client error: {exc}")

with st.sidebar:
    if st.button("🗑 Clear chat"):
        st.session_state["chat_msgs"] = []
        st.rerun()

# ------------------------------------------------------------------ page
st.title("💬 MedSupply Assistant")
if llm:
    st.caption(f"Groq configured · {llm.model}; access is checked when you send a message.")
else:
    st.caption("Offline · GROQ_API_KEY is not loaded from local secrets or the environment.")
st.caption("Ask about shortage status, stock coverage, suppliers, affected synthetic patients or the guidelines. "
           "Arabic and English. Answers use only the knowledge base (including anything you uploaded) and list their sources.")
st.warning("Operational supply information only — no prescribing, no dosing, no therapy switching.")

msgs = st.session_state.setdefault("chat_msgs", [])

EXAMPLES = [
    "ما هي الأدوية الموجودة وحالة كل دواء؟",
    "Which drugs are in shortage and how many days of stock are left?",
    "كام يوم مخزون Ceftriaxone؟ ومين الموردين المتاحين؟",
    "What do the guidelines say about a Ceftriaxone shortage?",
    "كم عدد المرضى (synthetic) المتأثرين بكل دواء؟",
]
if not msgs:
    st.caption("Try one of these:")
    cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if cols[i].button(ex, key=f"ex_{i}"):
            st.session_state["chat_pending"] = ex
            st.rerun()

for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + "; ".join(m["sources"]))

prompt = st.chat_input("اسأل عن الأدوية... / Ask about the drugs...")
prompt = prompt or st.session_state.pop("chat_pending", None)

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    # short memory: last user/assistant pairs
    pairs, pending = [], None
    for m in msgs:
        if m["role"] == "user":
            pending = m["content"]
        elif pending is not None:
            pairs.append((pending, m["content"]))
            pending = None

    # the workflow report from the main page (if a run has finished in this session)
    final = (st.session_state.get("wf") or {}).get("final") or {}
    report = final.get("final_report")

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                out = ce.ask(prompt, llm=llm, history_pairs=pairs, latest_report=report)
            except FileNotFoundError as exc:
                out = {"answer": f"⚠️ Dataset problem: {exc}", "sources": [], "context": ""}
            except Exception as exc:
                out = {"answer": f"⚠️ {type(exc).__name__}: {exc}", "sources": [], "context": ""}
        st.markdown(out["answer"])
        if out["sources"]:
            st.caption("Sources: " + "; ".join(out["sources"]))
        if out["context"]:
            with st.expander("Retrieved context"):
                st.code(out["context"])

    msgs.append({"role": "user", "content": prompt})
    msgs.append({"role": "assistant", "content": out["answer"], "sources": out["sources"]})
