"""
chat_engine.py — MedSupply Assistant (no Streamlit code here).

Ported from notebook section 17B. Differences from the notebook:
  * LLM = Groq API (same as the rest of the deployment) instead of local Mistral.
  * Reads the SAME merged CSVs / guideline PDFs that the main workflow uses (via kb.py),
    so an upload is visible to both the workflow and the chatbot.
    * Works without an API key too: it formats retrieved facts into a concise offline answer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import kb
import sentinel_core as core

# Arabic / short spellings -> a fragment of the English drug name. Extend freely.
DRUG_ALIASES = {
    "سيفترياكسون": "ceftriaxone", "سيفترايكسون": "ceftriaxone", "سفترياكسون": "ceftriaxone",
    "فانكومايسين": "vancomycin", "فانكوميسين": "vancomycin",
    "انسولين": "insulin", "جلارجين": "glargine", "لانتوس": "glargine",
    "اموكسيسيلين": "amoxicillin", "اموكسيلين": "amoxicillin",
}
MAX_CONTEXT_CHARS = 9000
SOURCES_MARK = "Sources: "

CHAT_SYSTEM_PROMPT = """You are MedSupply Assistant, a hospital-pharmacy SUPPLY assistant.
You answer questions about the drugs in the knowledge base: shortage status, inventory and coverage days,
suppliers, affected synthetic patients, and the guideline / uploaded documents.

Rules:
- Use ONLY the CONTEXT provided. Never use outside knowledge. Never invent numbers, dates, suppliers or documents.
- If the answer is not in the CONTEXT, say so clearly and say which data could be uploaded to answer it.
- Reply in the SAME language as the user's question (Arabic or English). Keep drug names, IDs and numbers exactly as written in the CONTEXT.
- coverage_days and urgency are already calculated in the CONTEXT: quote them, do not recalculate.
- Do NOT prescribe, diagnose, give doses, or recommend switching a patient's therapy. Supplier or stock options are operational
  information only and any alternative needs qualified pharmacist review.
- Be concise. Use short bullet points when listing several drugs."""


# ------------------------------------------------------------------ small helpers
def _norm_ar(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\u064B-\u0652\u0640]", "", s)                       # tashkeel + tatweel
    s = re.sub(r"[أإآ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return s


def _urgency_label(coverage: float, stock: int, reorder: int) -> str:
    """Same rule as the Inventory agent in sentinel_core."""
    return ("CRITICAL" if coverage < 2 else
            "HIGH" if coverage < 4 or stock <= reorder else
            "MEDIUM" if coverage < 7 else "LOW")


def _rows_ci(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    return df[df[col].astype(str).str.lower() == name.lower()]


def _latest_by_date(rows: pd.DataFrame, date_col: str = "reported_date") -> pd.Series:
    tmp = rows.assign(_d=pd.to_datetime(rows[date_col], errors="coerce")).sort_values(
        "_d", kind="stable", na_position="first")
    return tmp.iloc[-1].drop("_d")


# ------------------------------------------------------------------ retriever
class Retriever:
    """Builds 'drug cards' + document chunks from the current data and retrieves context for a question."""

    def __init__(self):
        self.cards: List[Dict[str, str]] = []
        self.doc_chunks: List[Dict[str, str]] = []
        self.overview = ""
        self.vec = self.mat = None
        self._build()

    def _build(self):
        t = kb.read_tables()
        drugs: Dict[str, Dict[str, str]] = {}                          # lower name -> {"name","id"}

        def add(name, did=""):
            n = str(name).strip()
            if n:
                d = drugs.setdefault(n.lower(), {"name": n, "id": ""})
                if did and not d["id"]:
                    d["id"] = str(did)

        for tb, ncol, icol in (("medications", "drug_name", "drug_id"), ("inventory", "drug_name", "drug_id"),
                               ("shortage_events", "drug_name", "drug_id"), ("suppliers", "medication", None),
                               ("patients", "medication", None)):
            for r in t[tb].to_dict("records"):
                add(r[ncol], r[icol] if icol else "")

        cards, overview = [], []
        for key in sorted(drugs):
            name, did = drugs[key]["name"], drugs[key]["id"]
            lines = [f"DRUG CARD — {name}" + (f" (drug_id {did})" if did else "")]
            med = _rows_ci(t["medications"], "drug_name", name)
            if not med.empty:
                lines.append(f"Category: {med.iloc[0]['category']}; catalog tag: {med.iloc[0]['demo_status']}")
            ev = _rows_ci(t["shortage_events"], "drug_name", name)
            status, sev = "NO EVENT RECORDED", ""
            if not ev.empty:
                e = _latest_by_date(ev)
                status, sev = e["shortage_status"], e["severity"]
                lines.append(f"Shortage status: {status} (severity {sev}/100), reported {e['reported_date']}, "
                             f"source {e['source']}" + (f", notes: {e['notes']}" if str(e["notes"]).strip() else ""))
            inv = _rows_ci(t["inventory"], "drug_name", name)
            stock_txt, urg = "no inventory record", ""
            if not inv.empty:
                i = inv.iloc[0]
                cov = float(i["current_stock"]) / max(float(i["daily_usage"]), 1e-9)
                urg = _urgency_label(cov, int(i["current_stock"]), int(i["reorder_level"]))
                stock_txt = f"stock={int(i['current_stock'])}, coverage={cov:.1f}d"
                lines.append(f"Inventory: stock {int(i['current_stock'])} units, daily usage {float(i['daily_usage']):g}, "
                             f"coverage_days {cov:.2f}, reorder level {int(i['reorder_level'])}, urgency {urg}, "
                             f"supplier_id {i['supplier_id']}, unit price {float(i['unit_price']):g}, "
                             f"last restock {i['last_restock_date']}")
            sup = _rows_ci(t["suppliers"], "medication", name)
            n_active = int((sup["supplier_status"] == "ACTIVE").sum()) if not sup.empty else 0
            if not sup.empty:
                lines.append("Suppliers: " + "; ".join(
                    f"{r.supplier_name} ({r.supplier_id}, {r.supplier_status}) qty {r.available_quantity}, "
                    f"price {r.unit_price:g}, lead time {r.lead_time_days}d"
                    for r in sup.sort_values("unit_price").head(6).itertuples()))
            pat = _rows_ci(t["patients"], "medication", name)
            pat = pat[pat["active_status"] == 1]
            n_pat = len(pat)
            if n_pat:
                cats = ", ".join(f"{k}: {v}" for k, v in pat["condition_category"].value_counts().items())
                lines.append(f"Active synthetic patients: {n_pat} (HIGH risk: {int((pat['risk_group'] == 'HIGH').sum())}); "
                             f"categories: {cats}")
            cards.append({"drug": name, "id": did, "text": "\n".join(lines), "label": f"Drug card: {name}"})
            overview.append(f"- {name} ({did or 'n/a'}): shortage={status}; {stock_txt}; urgency={urg or 'n/a'}; "
                            f"active_suppliers={n_active}; active_synthetic_patients={n_pat}")
        self.cards = cards
        self.overview = "\n".join(overview) if overview else "(no drugs in the knowledge base)"

        chunks = []
        for origin, f in kb.guideline_files():
            try:
                for num, text in kb.read_pdf_pages(f):
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        chunks.append({"text": text[:1500], "label": f"{origin}/{f.name} p.{num}"})
            except Exception:
                pass
        self.doc_chunks = chunks

        corpus = [c["text"] for c in cards] + [c["text"] for c in chunks]
        if corpus:
            self.vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
            self.mat = self.vec.fit_transform(corpus)

    def mentioned_drugs(self, question: str):
        q = _norm_ar(question)
        ids = set(re.findall(r"\bm\d{3,}\b", q))
        alias = {_norm_ar(k): v for k, v in DRUG_ALIASES.items()}
        hits = []
        for c in self.cards:
            nm = _norm_ar(c["drug"])
            words = [w for w in re.findall(r"[a-z0-9]+", nm) if len(w) >= 5]
            if (nm in q or (c["id"] and c["id"].lower() in ids)
                    or any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)
                    or any(a in q and frag in nm for a, frag in alias.items())):
                hits.append(c)
        return hits

    def retrieve(self, question: str, latest_report: Optional[str] = None, top_docs: int = 4) -> Tuple[str, List[str]]:
        """Return (context_text, source_labels)."""
        mentioned = self.mentioned_drugs(question)
        parts = ["OVERVIEW OF ALL DRUGS IN THE KNOWLEDGE BASE:\n" + self.overview]
        sources = ["Knowledge base overview"]

        doc_hits: List[Dict[str, str]] = []
        if self.vec is not None:
            qtext = question + " " + " ".join(c["drug"] for c in mentioned)
            sims = linear_kernel(self.vec.transform([qtext]), self.mat).ravel()
            n_cards = len(self.cards)
            if not mentioned:                                          # nothing named -> best matching cards
                best = sorted(range(n_cards), key=lambda i: -sims[i])[:2]
                mentioned = [self.cards[i] for i in best if sims[i] > 0.1]
            doc_rank = sorted(range(len(self.doc_chunks)), key=lambda i: -sims[n_cards + i])
            doc_hits = [self.doc_chunks[i] for i in doc_rank[:top_docs] if sims[n_cards + i] > 0.05]

        for c in mentioned[:3]:
            parts.append(c["text"])
            sources.append(c["label"])
        for d in doc_hits:
            parts.append(f"[{d['label']}]\n{d['text']}")
            sources.append(d["label"])

        wants_report = any(w in _norm_ar(question) for w in
                           ("report", "workflow", "purchase", "approval", "تقرير", "شراء", "الموافقه", "موافقه"))
        if wants_report and latest_report:
            parts.append("LATEST WORKFLOW REPORT:\n" + latest_report[:2500])
            sources.append("Latest workflow report")

        return "\n\n".join(parts)[:MAX_CONTEXT_CHARS], sources


_STATE: Dict[str, Any] = {"fp": None, "ret": None}


def get_retriever() -> Retriever:
    """Rebuilt automatically whenever an upload changes a table or a guideline document."""
    fp = kb.data_fingerprint()
    if _STATE["fp"] != fp or _STATE["ret"] is None:
        _STATE["ret"] = Retriever()
        _STATE["fp"] = fp
    return _STATE["ret"]


# ------------------------------------------------------------------ LLM
class GroqChat:
    """Groq chat client with a fallback model (same models as the rest of the app)."""

    def __init__(self, api_key: str, model: str = core.GROQ_DEFAULT_MODEL,
                 fallback_model: str = core.GROQ_FALLBACK_MODEL):
        from groq import Groq
        self.client = Groq(api_key=api_key, max_retries=2, timeout=30.0)
        self.model, self.fallback_model = model, fallback_model
        self.last_model_used: Optional[str] = None
        self.last_error: Optional[str] = None

    def chat(self, question: str, context: str, history_pairs=None, max_tokens: int = 500) -> str:
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        for u, a in (history_pairs or [])[-3:]:                        # short memory of the last turns
            messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"})
        models = [self.model] + ([self.fallback_model] if self.fallback_model and self.fallback_model != self.model else [])
        for m in models:
            try:
                resp = self.client.chat.completions.create(
                    model=m, messages=messages, temperature=0.2, max_tokens=max_tokens, top_p=0.9)
                self.last_model_used, self.last_error = m, None
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:                                    # rate limit, network, bad model id ...
                self.last_error = f"{type(exc).__name__}: {exc}"
        return f"⚠️ The model could not answer ({self.last_error})"


def ask(question: str, llm: Optional[GroqChat] = None, history_pairs=None,
        latest_report: Optional[str] = None) -> Dict[str, Any]:
    """Answer one question. Returns {"answer", "sources", "context"}."""
    question = (question or "").strip()
    if not question:
        return {"answer": "اكتب سؤالك عن الأدوية / Type a question about the drugs.", "sources": [], "context": ""}
    retriever = get_retriever()
    context, sources = retriever.retrieve(question, latest_report)
    if llm is None:
        answer = _offline_answer(question, context, retriever)
    else:
        answer = llm.chat(question, context, history_pairs)
    return {"answer": answer, "sources": list(dict.fromkeys(sources)), "context": context}


def _offline_answer(question: str, context: str, retriever: Retriever) -> str:
    """Turn the retrieved drug overview into a short, grounded answer without an LLM."""
    row_pattern = re.compile(
        r"^- (?P<drug>.+?) \((?P<drug_id>[^)]+)\): shortage=(?P<shortage>[^;]+); "
        r"stock=(?P<stock>[^,]+), coverage=(?P<coverage>[^;]+); urgency=(?P<urgency>[^;]+); "
        r"active_suppliers=(?P<suppliers>\d+); active_synthetic_patients=(?P<patients>\d+)$"
    )
    rows = [match.groupdict() for line in context.splitlines()
            if (match := row_pattern.match(line.strip()))]
    if not rows:
        return "I couldn't find structured drug overview data for this question."

    mentioned_ids = {item["id"] for item in retriever.mentioned_drugs(question) if item.get("id")}
    if mentioned_ids:
        selected_rows = [row for row in rows if row["drug_id"] in mentioned_ids]
        if selected_rows:
            rows = selected_rows

    is_arabic = bool(re.search(r"[\u0600-\u06ff]", question))
    if is_arabic:
        lines = ["بحسب البيانات الحالية، هذا ملخص حالة الإمداد:"]
        for row in rows:
            status = row["shortage"].replace("_", " ")
            coverage = re.sub(r"d$", "", row["coverage"].strip())
            lines.append(
                f"- {row['drug']} ({row['drug_id']}): الحالة {status}، المخزون {row['stock']} وحدة "
                f"يكفي لنحو {coverage} يومًا، ومستوى الإلحاح {row['urgency']}. "
                f"الموردون النشطون: {row['suppliers']}؛ المرضى الافتراضيون النشطون: {row['patients']}."
            )
    else:
        lines = ["Based on the current records, here is the supply summary:"]
        for row in rows:
            status = row["shortage"].replace("_", " ").lower()
            coverage = re.sub(r"d$", "", row["coverage"].strip())
            supplier_word = "supplier" if row["suppliers"] == "1" else "suppliers"
            patient_word = "patient" if row["patients"] == "1" else "patients"
            lines.append(
                f"- {row['drug']} ({row['drug_id']}) is marked {status}, with {row['stock']} units "
                f"(about {coverage} days of stock) and {row['urgency'].lower()} urgency. "
                f"The dataset lists {row['suppliers']} active {supplier_word} and "
                f"{row['patients']} active synthetic {patient_word}."
            )
    return "\n\n".join(lines)
