"""📤 Upload data — add / update tables and guideline documents (notebook 3B + 17C, tab 1)."""
import streamlit as st

import kb
import rag_store


st.set_page_config(
    page_title="Upload Data · MedSupply Sentinel",
    page_icon="📤",
    layout="wide",
)


st.title("📤 Upload data")

st.caption(
    "Update inventory, suppliers, shortage events, medications, "
    "synthetic patients, or add guideline documents. The change is "
    "used by the **main workflow page** and by the **chat** on their next run."
)

st.warning(
    "Synthetic patient data only — rows whose `patient_id` "
    "does not start with `SYN-` are refused."
)


with st.expander(
    "What can I upload?",
    expanded=False,
):
    st.markdown(
        kb.UPLOAD_HELP
    )


# the uploader is re-created after each ingest
st.session_state.setdefault(
    "kb_uploader_n",
    0,
)

st.session_state.setdefault(
    "kb_last_report",
    "",
)


files = st.file_uploader(
    "Drop files here",
    accept_multiple_files=True,
    type=kb.ACCEPTED_TYPES,
    key=(
        f"kb_uploader_"
        f"{st.session_state['kb_uploader_n']}"
    ),
)


c1, c2, _ = st.columns(
    [1, 1, 3]
)


if c1.button(
    "➕ Add to knowledge base",
    type="primary",
    disabled=not files,
):
    with st.spinner(
        "Validating, merging, and rebuilding the RAG index…"
    ):
        reports = [
            kb.ingest_bytes(
                f.name,
                f.getvalue(),
            )
            for f in files
        ]

        try:
            rag_report = (
                rag_store
                .get_kb()
                .sync(force=True)
            )

            rag_status = (
                f"RAG index: "
                f"{rag_report.summary()}"
            )

            if rag_report.warnings:
                rag_status += (
                    "\n\nWarnings:\n"
                    + "\n".join(
                        f"- {w}"
                        for w in rag_report.warnings
                    )
                )

        except Exception as exc:
            rag_status = (
                "⚠️ RAG index rebuild failed: "
                f"{type(exc).__name__}: {exc}. "
                "The uploaded data itself was still "
                "processed by kb.py."
            )

    st.session_state["kb_last_report"] = (
        kb.format_reports(reports)
        + "\n\n"
        + rag_status
    )

    st.session_state[
        "kb_uploader_n"
    ] += 1

    st.rerun()


confirm = c2.checkbox(
    "Confirm reset"
)


if c2.button(
    "♻️ Reset to original data",
    disabled=not confirm,
):
    with st.spinner(
        "Resetting data and rebuilding the RAG index…"
    ):
        kb.reset()

        try:
            # Drop the in-memory singleton first.
            rag_store.reset_singleton()

            rag_report = (
                rag_store
                .get_kb()
                .sync(force=True)
            )

            rag_status = (
                f"RAG index reset: "
                f"{rag_report.summary()}"
            )

        except Exception as exc:
            rag_status = (
                "⚠️ RAG index reset failed: "
                f"{type(exc).__name__}: {exc}"
            )

    st.session_state["kb_last_report"] = (
        "♻️ Everything reset to the original dataset."
        "\n\n"
        + rag_status
    )

    st.session_state[
        "kb_uploader_n"
    ] += 1

    st.rerun()


if st.session_state["kb_last_report"]:
    st.markdown(
        "### Result"
    )

    st.markdown(
        st.session_state["kb_last_report"]
    )

    try:
        st.page_link(
            "app.py",
            label="Go to the main workflow page →",
            icon="💊",
        )
    except Exception:
        pass


st.divider()


try:
    st.subheader(
        "Tables (rows now vs. original dataset)"
    )

    st.dataframe(
        kb.status(),
        hide_index=True,
    )

    st.subheader(
        "Guideline / reference documents"
    )

    st.dataframe(
        kb.documents(),
        hide_index=True,
    )

except Exception as exc:
    st.error(
        f"Could not read the dataset: {exc}"
    )


with st.expander(
    "Try it with sample files (adds a new drug 'Amoxicillin', M004)"
):
    st.write(
        "The set contains medications, inventory "
        "(also updates M002's stock), shortage event, "
        "supplier, two synthetic patients and a guideline note. "
        "After loading, pick **Amoxicillin (M004)** on the "
        "main page. *Reset* removes everything again."
    )

    cols = st.columns(3)

    for i, (fname, content) in enumerate(
        kb.SAMPLE_FILES.items()
    ):
        cols[i % 3].download_button(
            f"⬇ {fname}",
            content,
            file_name=fname,
            key=f"dl_{fname}",
        )

    if st.button(
        "Load all sample files now"
    ):
        with st.spinner(
            "Loading samples and rebuilding the RAG index…"
        ):
            sample_reports = (
                kb.ingest_samples()
            )

            try:
                rag_report = (
                    rag_store
                    .get_kb()
                    .sync(force=True)
                )

                rag_status = (
                    f"RAG index: "
                    f"{rag_report.summary()}"
                )

            except Exception as exc:
                rag_status = (
                    "⚠️ RAG index rebuild failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            st.session_state[
                "kb_last_report"
            ] = (
                kb.format_reports(
                    sample_reports
                )
                + "\n\n"
                + rag_status
            )

        st.rerun()


with st.expander(
    "Upload history"
):
    hist = kb.history()

    if not hist:
        st.caption(
            "Nothing uploaded yet."
        )

    for h in reversed(
        hist[-15:]
    ):
        st.markdown(
            f"`{h.get('time', '')[:19]}` — "
            f"{kb.format_reports([h])}"
        )