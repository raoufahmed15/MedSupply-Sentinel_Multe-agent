"""
chat_engine.py — MedSupply Assistant (no Streamlit code here).

Architecture:
    Uploaded / dataset data
        -> kb.py / stored CSV + guideline files
        -> rag_store.py / unified RAG index
        -> this module / retrieval + Groq answer generation
        -> 2_Chat.py / Streamlit UI

The old, duplicated Retriever implementation has been removed. The single
retrieval source is now rag_store.py.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import rag_store as rag
import sentinel_core as core


# ------------------------------------------------------------------ config
MAX_CONTEXT_CHARS = 12000

# Arabic / short spellings -> a fragment of the English drug name.
# Extend freely when the dataset contains additional common spellings.
DRUG_ALIASES = {
    "سيفترياكسون": "ceftriaxone",
    "سيفترايكسون": "ceftriaxone",
    "سفترياكسون": "ceftriaxone",
    "فانكومايسين": "vancomycin",
    "فانكوميسين": "vancomycin",
    "انسولين": "insulin",
    "جلارجين": "glargine",
    "لانتوس": "glargine",
    "اموكسيسيلين": "amoxicillin",
    "اموكسيلين": "amoxicillin",
}

ARABIC_QUERY_ALIASES = {
    "مورد": "supplier record supplier directory",
    "بنشتري": "each supplier what medication",
    "نشتري": "each supplier what medication",
    "شراء": "supplier medication",
    "مشتريات": "supplier medication",
    "صنف": "product item medication",
    "منتج": "product item medication",
    "دواء": "drug medication",
    "ادويه": "drugs medications",
    "مريض": "patient record",
    "مرض": "patient record",
    "ارشاد": "guideline guidance",
    "دليل": "guideline guidance",
    "ملف": "uploaded document",
    "مستند": "uploaded document",
    "بيانات": "data dataset table",
}


CHAT_SYSTEM_PROMPT = """You are MedSupply Assistant, a hospital-pharmacy SUPPLY assistant.
You answer questions about the drugs in the knowledge base: shortage status, inventory and coverage days,
suppliers, affected synthetic patients, workflow results, and guideline / uploaded documents.

Rules:
- Use ONLY the CONTEXT provided. Never use outside knowledge.
- Never invent numbers, dates, suppliers, documents, workflow results, or patient counts.
- If the answer is not in the CONTEXT, say so clearly and state what data would be needed.
- Reply in the SAME language as the user's question (Arabic or English).
- Keep drug names, IDs, dates, and numbers exactly as written in the CONTEXT.
- Treat raw uploaded/data records as facts. Treat workflow outputs as AI/derived analysis and mention that
  human/pharmacist review is required when relevant.
- coverage_days and urgency are already calculated in the CONTEXT: quote them; do not recalculate them.
- Do NOT prescribe, diagnose, give doses, or recommend switching a patient's therapy.
- Supplier or stock options are operational information only. Any alternative requires qualified pharmacist review.
- Be concise. Use short bullet points when listing several items."""


# ------------------------------------------------------------------ text helpers
def _norm_ar(s: Any) -> str:
    return rag.norm_text(s)


def _expand_query(question: str) -> str:
    q = _norm_ar(question)
    aliases = dict.fromkeys(
        expansion
        for term, expansion in ARABIC_QUERY_ALIASES.items()
        if _norm_ar(term) in q
    )
    return " ".join([question, *aliases])


def _is_report_question(question: str) -> bool:
    q = _norm_ar(question)
    return any(
        token in q
        for token in (
            "report",
            "workflow",
            "purchase",
            "approval",
            "incident",
            "تقرير",
            "شراء",
            "الموافقه",
            "موافقه",
            "موافقة",
            "موافقت",
            "امر شراء",
        )
    )


def _asks_for_non_med_data(question: str) -> bool:
    q = _norm_ar(question)
    topics = (
        "patient", "supplier", "vendor", "guideline", "document", "uploaded",
        "reference", "workflow", "approval", "purchase", "dataset", "database",
        "knowledge base", "records", "files", "tables", "مريض", "مرض", "مورد",
        "دليل", "ارشاد", "مستند", "وثيقه", "ملف", "مرجع", "بيانات", "سجل",
        "جدول", "قاعده المعرفه", "قاعده البيانات", "مرفق", "شراء", "موافقه",
        "سير العمل",
    )
    return any(topic in q for topic in topics)


def _mentioned_drugs(question: str, kb: rag.KnowledgeBase) -> List[dict]:
    """Resolve explicit medication mentions using the unified RAG entity registry."""
    q = _norm_ar(question)
    entities = kb.entities().get("drugs", [])
    hits: List[dict] = []

    alias_map = {_norm_ar(k): _norm_ar(v) for k, v in DRUG_ALIASES.items()}

    for drug in entities:
        variants = set(drug.get("variants") or [])
        variants.add(drug.get("name") or "")
        variants.add(drug.get("key") or "")

        matched = False
        for variant in variants:
            nv = _norm_ar(variant)
            if len(nv) >= 4 and nv in q:
                matched = True
                break

        did = _norm_ar(drug.get("id") or "")
        if not matched and did and re.search(rf"\b{re.escape(did)}\b", q):
            matched = True

        if not matched:
            for arabic_alias, english_fragment in alias_map.items():
                if arabic_alias in q and english_fragment in _norm_ar(drug.get("name") or ""):
                    matched = True
                    break

        if matched:
            hits.append(drug)

    return hits


def _dedupe_docs(docs: List[rag.Doc]) -> List[rag.Doc]:
    out: List[rag.Doc] = []
    seen = set()

    for doc in docs:
        if doc.doc_id in seen:
            continue

        seen.add(doc.doc_id)
        out.append(doc)

    return out


# ------------------------------------------------------------------ retrieval
class Retriever:
    """Thin compatibility wrapper around the unified rag_store KnowledgeBase."""

    def __init__(self):
        self.kb = rag.get_kb()
        self.sync_report = self.kb.sync_if_needed()

    def refresh(self) -> None:
        self.kb = rag.get_kb()
        self.sync_report = self.kb.sync_if_needed()

    def mentioned_drugs(self, question: str) -> List[dict]:
        self.refresh()
        return _mentioned_drugs(question, self.kb)

    def retrieve(
        self,
        question: str,
        latest_report: Optional[str] = None,
        top_docs: int = 8,
    ) -> Tuple[str, List[str]]:
        """Return grounded context text and source labels from the unified RAG index."""
        self.refresh()

        question = (question or "").strip()

        if not question:
            return "", []

        mentioned = self.mentioned_drugs(question)
        mentioned_keys = [d.get("key") for d in mentioned if d.get("key")]
        wants_report = _is_report_question(question)

        docs: List[rag.Doc] = []

        # Keep the global drug matrix for medication overviews, not unrelated data queries.
        if not _asks_for_non_med_data(question):
            docs.extend(self.kb.find(document_type="overview"))

        # 2) If the user names a medication, include its joined profile and
        #    patient-impact summary directly.
        for key in mentioned_keys:
            docs.extend(
                self.kb.find(
                    document_type="medication_profile",
                    medication_keys=key,
                )
            )
            docs.extend(
                self.kb.find(
                    document_type="patient_impact_summary",
                    medication_keys=key,
                )
            )

        # 3) If the question is about workflow / approval / purchase, include
        #    persisted workflow analysis for the mentioned medication(s).
        if wants_report:
            if mentioned_keys:
                for key in mentioned_keys:
                    docs.extend(
                        self.kb.find(
                            medication_keys=key,
                            source_type=rag.SRC_WORKFLOW,
                        )
                    )
            else:
                docs.extend(
                    self.kb.find(
                        source_type=rag.SRC_WORKFLOW,
                    )
                )

        # 4) Semantic retrieval over the unified index.
        scores = self.kb.semantic(_expand_query(question))

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        added_semantic = 0

        for doc_id, score in ranked:
            if score <= 0:
                continue

            doc = self.kb.get(doc_id)

            if doc is None:
                continue

            if doc in docs:
                continue

            docs.append(doc)
            added_semantic += 1

            if added_semantic >= top_docs:
                break

        # 5) Keep current-session report support as a convenience even if the
        #    persisted workflow snapshot is not yet available.
        if wants_report and latest_report:
            latest_report = latest_report.strip()

            if latest_report:
                docs_text = (
                    "LATEST WORKFLOW REPORT (current Streamlit session):\n"
                    + latest_report[:3500]
                )
            else:
                docs_text = ""
        else:
            docs_text = ""

        docs = _dedupe_docs(docs)

        # Keep the context bounded.
        parts: List[str] = []
        sources: List[str] = []
        used_chars = 0

        priority_types = {
            "overview": 10,
            "medication_profile": 20,
            "patient_impact_summary": 30,
            "workflow_shortage_analysis": 40,
            "workflow_inventory_analysis": 41,
            "workflow_guideline_analysis": 42,
            "workflow_patient_impact": 43,
            "workflow_substitution_analysis": 44,
            "workflow_validation": 45,
            "workflow_purchase_order": 46,
            "workflow_report_1": 47,
            "workflow_report_2": 48,
            "workflow_report_3": 49,
        }

        docs.sort(
            key=lambda d: priority_types.get(
                d.meta.get("document_type"),
                100,
            )
        )

        for doc in docs:
            block = f"[{doc.label}]\n{doc.text}"

            remaining = MAX_CONTEXT_CHARS - used_chars

            if remaining <= 0:
                break

            if len(block) > remaining:
                block = block[:remaining]

            parts.append(block)
            sources.append(doc.label)
            used_chars += len(block) + 2

        if docs_text and used_chars < MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - used_chars

            parts.append(docs_text[:remaining])
            sources.append("Latest workflow report (current session)")

        if not parts:
            parts = [
                "No relevant knowledge-base records were retrieved for this question."
            ]
            sources = []

        return (
            "\n\n".join(parts)[:MAX_CONTEXT_CHARS],
            list(dict.fromkeys(sources)),
        )


_STATE: Dict[str, Any] = {
    "fp": None,
    "ret": None,
}


def get_retriever() -> Retriever:
    """Reuse the retriever object while rag_store itself handles source changes."""
    kb = rag.get_kb()

    kb.sync_if_needed()

    fp = kb._fp

    if _STATE["ret"] is None or _STATE["fp"] != fp:
        _STATE["ret"] = Retriever()
        _STATE["fp"] = fp
    else:
        _STATE["ret"].refresh()

    return _STATE["ret"]


# ------------------------------------------------------------------ LLM
class GroqChat:
    """Groq chat client with a fallback model."""

    def __init__(
        self,
        api_key: str,
        model: str = core.GROQ_DEFAULT_MODEL,
        fallback_model: str = core.GROQ_FALLBACK_MODEL,
    ):
        from groq import Groq

        self.client = Groq(
            api_key=api_key,
            max_retries=2,
            timeout=30.0,
        )

        self.model = model
        self.fallback_model = fallback_model

        self.last_model_used: Optional[str] = None
        self.last_error: Optional[str] = None

    def chat(
        self,
        question: str,
        context: str,
        history_pairs=None,
        max_tokens: int = 1200,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": CHAT_SYSTEM_PROMPT,
            }
        ]

        for user_text, assistant_text in (history_pairs or [])[-3:]:
            messages += [
                {
                    "role": "user",
                    "content": user_text,
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ]

        messages.append(
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}"
                ),
            }
        )

        models = [self.model]

        if (
            self.fallback_model
            and self.fallback_model != self.model
        ):
            models.append(self.fallback_model)

        for model_name in models:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    top_p=0.9,
                )

                self.last_model_used = model_name
                self.last_error = None

                return (
                    response.choices[0].message.content or ""
                ).strip()

            except Exception as exc:
                self.last_error = (
                    f"{type(exc).__name__}: {exc}"
                )

        return (
            f"⚠️ The model could not answer "
            f"({self.last_error})"
        )


# ------------------------------------------------------------------ public answer API
def ask(
    question: str,
    llm: Optional[GroqChat] = None,
    history_pairs=None,
    latest_report: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer one question. Returns {"answer", "sources", "context"}."""
    question = (question or "").strip()

    if not question:
        return {
            "answer": (
                "اكتب سؤالك عن الأدوية / "
                "Type a question about the drugs."
            ),
            "sources": [],
            "context": "",
        }

    retriever = get_retriever()

    context, sources = retriever.retrieve(
        question,
        latest_report=latest_report,
    )

    if llm is None:
        answer = _offline_answer(
            question,
            context,
            retriever,
        )
    else:
        answer = llm.chat(
            question,
            context,
            history_pairs,
        )

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "context": context,
    }


# ------------------------------------------------------------------ offline fallback
def _offline_answer(
    question: str,
    context: str,
    retriever: Retriever,
) -> str:
    """Provide a concise grounded answer when Groq is unavailable."""

    row_pattern = re.compile(
        r"^- (?P<drug>.+?) \((?P<drug_id>[^)]+)\): "
        r"shortage=(?P<shortage>[^;]+); "
        r"stock=(?P<stock>[^,]+), "
        r"coverage=(?P<coverage>[^;]+); "
        r"urgency=(?P<urgency>[^;]+); "
        r"active_suppliers=(?P<suppliers>\d+); "
        r"active_synthetic_patients=(?P<patients>\d+)$"
    )

    rows = [
        match.groupdict()
        for line in context.splitlines()
        if (match := row_pattern.match(line.strip()))
    ]

    mentioned = retriever.mentioned_drugs(question)

    mentioned_ids = {
        str(item.get("id") or "")
        for item in mentioned
        if item.get("id")
    }

    if mentioned_ids:
        selected_rows = [
            row
            for row in rows
            if row["drug_id"] in mentioned_ids
        ]

        if selected_rows:
            rows = selected_rows

    if rows:
        is_arabic = bool(
            re.search(r"[\u0600-\u06ff]", question)
        )

        if is_arabic:
            lines = [
                "بحسب البيانات الحالية، هذا ملخص حالة الإمداد:"
            ]

            for row in rows:
                status = row["shortage"].replace("_", " ")
                coverage = re.sub(
                    r"d$",
                    "",
                    row["coverage"].strip(),
                )

                lines.append(
                    f"- {row['drug']} ({row['drug_id']}): "
                    f"الحالة {status}، "
                    f"المخزون {row['stock']} وحدة، "
                    f"يكفي لنحو {coverage} يومًا، "
                    f"ومستوى الإلحاح {row['urgency']}. "
                    f"الموردون النشطون: {row['suppliers']}؛ "
                    f"المرضى الافتراضيون النشطون: "
                    f"{row['patients']}."
                )

            return "\n\n".join(lines)

        lines = [
            "Based on the current records, "
            "here is the supply summary:"
        ]

        for row in rows:
            status = row["shortage"].replace(
                "_",
                " ",
            ).lower()

            coverage = re.sub(
                r"d$",
                "",
                row["coverage"].strip(),
            )

            supplier_word = (
                "supplier"
                if row["suppliers"] == "1"
                else "suppliers"
            )

            patient_word = (
                "patient"
                if row["patients"] == "1"
                else "patients"
            )

            lines.append(
                f"- {row['drug']} ({row['drug_id']}) "
                f"is marked {status}, "
                f"with {row['stock']} units "
                f"(about {coverage} days of stock) "
                f"and {row['urgency'].lower()} urgency. "
                f"The dataset lists "
                f"{row['suppliers']} active "
                f"{supplier_word} and "
                f"{row['patients']} active synthetic "
                f"{patient_word}."
            )

        return "\n\n".join(lines)

    if _is_report_question(question):
        return (
            "No structured overview row was found for this question. "
            "The retrieved context may still contain workflow/document details."
        )

    return (
        "I couldn't find structured drug overview data "
        "for this question."
    )