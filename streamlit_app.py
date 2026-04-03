"""
Leadership Insight Agent — Streamlit UI

Run from project root:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from agent import LeadershipInsightAgent

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS = PROJECT_ROOT / "documents"
DEFAULT_INDEX = PROJECT_ROOT / "vector_index_ui"

MODEL_CHOICES: Dict[str, List[str]] = {
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "o1-mini",
        "o3-mini",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-90b-vision-preview",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "ollama": [
        "llama3.2",
        "llama3.1",
        "mistral",
        "phi3",
        "qwen2.5",
    ],
}

PROVIDER_OPTIONS: List[Tuple[str, str, str]] = [
    ("Groq — free API tier", "groq", "Fast inference · console.groq.com"),
    ("Ollama — local / open-source", "ollama", "No API key · ollama.com"),
    ("Anthropic — Claude", "anthropic", "console.anthropic.com"),
    ("OpenAI", "openai", "platform.openai.com"),
]

CUSTOM_MODEL_LABEL = "Custom model…"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Fraunces:wght@500;600;700&display=swap');
            html, body, [data-testid="stAppViewContainer"] {
                font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif;
            }
            [data-testid="stAppViewContainer"] {
                background: radial-gradient(1000px 500px at 8% -8%, rgba(99, 102, 241, 0.16), transparent),
                    radial-gradient(700px 400px at 92% 0%, rgba(16, 185, 129, 0.1), transparent),
                    linear-gradient(165deg, #0c1222 0%, #111827 50%, #0f172a 100%);
            }
            section[data-testid="stSidebar"] > div {
                background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.12);
            }
            h1.hero-title {
                font-family: 'Fraunces', Georgia, serif;
                font-weight: 600;
                letter-spacing: -0.02em;
                background: linear-gradient(120deg, #f1f5f9 0%, #a5b4fc 45%, #6ee7b7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2rem;
                margin-bottom: 0.35rem;
            }
            .hero-sub { color: #94a3b8; font-size: 1.02rem; line-height: 1.55; max-width: 40rem; }
            [data-testid="stVerticalBlock"] > div:has(div[data-testid="stChatMessage"]) {
                gap: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _secrets_get(key: str) -> str:
    try:
        v = st.secrets[key]
        return (str(v) if v else "").strip()
    except Exception:
        return ""


def _resolve_model_id(provider: str, choice: str, custom: str) -> str:
    if choice == CUSTOM_MODEL_LABEL:
        c = (custom or "").strip()
        return c if c else MODEL_CHOICES[provider][0]
    return choice


def _save_uploads(files, dest: Path) -> str:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        (dest / f.name).write_bytes(f.getvalue())
    return str(dest.resolve())


def _build_agent(
    *,
    provider: str,
    model: str,
    top_k: int,
    docs_dir: str,
    index_path: str,
    api_key: str,
    ollama_base: str,
    groq_base: str,
) -> LeadershipInsightAgent:
    kwargs = dict(
        docs_dir=docs_dir,
        index_path=index_path,
        model=model,
        top_k=top_k,
        llm_provider=provider,
    )
    if provider == "anthropic":
        kwargs["anthropic_api_key"] = api_key
    elif provider == "openai":
        kwargs["openai_api_key"] = api_key
    elif provider == "groq":
        kwargs["groq_api_key"] = api_key
        if groq_base.strip():
            kwargs["groq_base_url"] = groq_base.strip()
    else:
        if ollama_base.strip():
            kwargs["ollama_base_url"] = ollama_base.strip()
    return LeadershipInsightAgent(**kwargs)


def main() -> None:
    st.set_page_config(
        page_title="Leadership Insight",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "ingest_ok" not in st.session_state:
        st.session_state.ingest_ok = False
    # --- Sidebar ------------------------------------------------------------
    with st.sidebar:
        st.markdown("### LLM & retrieval")
        st.caption("API keys are kept in this browser session only unless you use Streamlit Cloud secrets.")

        labels = [x[0] for x in PROVIDER_OPTIONS]
        keys = [x[1] for x in PROVIDER_OPTIONS]
        idx = st.selectbox(
            "Provider",
            range(len(PROVIDER_OPTIONS)),
            format_func=lambda i: labels[i],
            key="sb_provider_idx",
        )
        provider = keys[idx]
        st.caption(PROVIDER_OPTIONS[idx][2])

        api_key = ""
        if provider == "anthropic":
            api_key = st.text_input(
                "Anthropic API key",
                type="password",
                value=_secrets_get("anthropic_api_key"),
                key="sb_anthropic_key",
                autocomplete="off",
            )
        elif provider == "openai":
            api_key = st.text_input(
                "OpenAI API key",
                type="password",
                value=_secrets_get("openai_api_key"),
                key="sb_openai_key",
                autocomplete="off",
            )
        elif provider == "groq":
            api_key = st.text_input(
                "Groq API key",
                type="password",
                value=_secrets_get("groq_api_key"),
                key="sb_groq_key",
                placeholder="gsk_…",
                autocomplete="off",
            )
        else:
            st.text_input(
                "Ollama API base",
                value="http://localhost:11434/v1",
                key="sb_ollama_base",
                help="OpenAI-compatible `/v1` base URL for your Ollama instance.",
            )

        model_opts = MODEL_CHOICES[provider] + [CUSTOM_MODEL_LABEL]
        st.selectbox(
            "Model",
            model_opts,
            key="sb_model_pick",
        )
        if st.session_state.get("sb_model_pick") == CUSTOM_MODEL_LABEL:
            st.text_input(
                "Custom model id",
                key="sb_model_custom",
                placeholder="e.g. llama3.2:latest",
            )

        st.slider("Document chunks (top-k)", 3, 12, 6, key="sb_top_k")

        st.markdown("---")
        st.markdown("##### Knowledge base")
        docs_mode = st.radio(
            "Source",
            ["Bundled examples", "Project folder", "Upload files"],
            key="sb_docs_mode",
        )
        docs_dir_str = str(DEFAULT_DOCS)
        if docs_mode == "Project folder":
            docs_dir_str = st.text_input(
                "Documents folder",
                value=str(DEFAULT_DOCS),
                key="sb_docs_folder",
            )
        elif docs_mode == "Upload files":
            up = st.file_uploader(
                "Files",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="sb_upload",
            )
            if up and len(up) > 0:
                dest = Path(tempfile.gettempdir()) / "leadership_agent_docs"
                docs_dir_str = _save_uploads(up, dest)
            else:
                docs_dir_str = str(DEFAULT_DOCS)

        with st.expander("Advanced"):
            st.text_input(
                "Vector index path",
                value=str(DEFAULT_INDEX),
                key="sb_index_path",
            )
            st.checkbox(
                "Rebuild index on apply",
                key="sb_rebuild",
                help="Use after adding or replacing documents.",
            )
            if provider == "groq":
                st.text_input(
                    "Groq base URL (optional)",
                    value="",
                    key="sb_groq_base",
                    placeholder="https://api.groq.com/openai/v1",
                )

        apply = st.button("Apply & load documents", type="primary", use_container_width=True)

    groq_adv = (st.session_state.get("sb_groq_base") or "").strip()

    pick = st.session_state.get("sb_model_pick") or MODEL_CHOICES[provider][0]
    custom_part = (
        (st.session_state.get("sb_model_custom") or "").strip()
        if pick == CUSTOM_MODEL_LABEL
        else ""
    )
    resolved_model = _resolve_model_id(provider, pick, custom_part)
    index_path_val = st.session_state.get("sb_index_path", str(DEFAULT_INDEX))
    ollama_base = (st.session_state.get("sb_ollama_base") or "http://localhost:11434/v1").strip()

    # --- Apply --------------------------------------------------------------
    if apply:
        st.session_state.ingest_ok = False
        st.session_state.agent = None
        st.session_state.messages = []
        try:
            if docs_mode == "Upload files" and not st.session_state.get("sb_upload"):
                st.error("Please upload at least one document, or choose another source.")
            else:
                with st.spinner("Building index and connecting to the model…"):
                    agent = _build_agent(
                        provider=provider,
                        model=resolved_model,
                        top_k=int(st.session_state.get("sb_top_k", 6)),
                        docs_dir=docs_dir_str,
                        index_path=index_path_val,
                        api_key=api_key.strip(),
                        ollama_base=ollama_base,
                        groq_base=groq_adv,
                    )
                    agent.ingest_documents(force_rebuild=bool(st.session_state.get("sb_rebuild")))
                    st.session_state.agent = agent
                    st.session_state.ingest_ok = True
                st.sidebar.success("Ready — ask a question in the main panel.")
        except ValueError as e:
            st.sidebar.error(str(e))
        except Exception as e:
            st.sidebar.error(f"{type(e).__name__}: {e}")
            with st.sidebar.expander("Details"):
                st.code(traceback.format_exc())

    # --- Main ---------------------------------------------------------------
    st.markdown('<h1 class="hero-title">Leadership Insight</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Executive Q&amp;A grounded in your internal documents. '
        "Configure the model in the sidebar, load your knowledge base, then chat below.</p>",
        unsafe_allow_html=True,
    )

    agent = st.session_state.agent
    if agent and st.session_state.ingest_ok:
        pills = (
            f'<span class="hero-sub"><b>Provider</b> · {agent.llm_provider} &nbsp;|&nbsp; '
            f"<b>Model</b> · {agent.model} &nbsp;|&nbsp; <b>Top-k</b> · {agent.top_k}</span>"
        )
        st.markdown(pills, unsafe_allow_html=True)

    if not st.session_state.ingest_ok or agent is None:
        st.info(
            "👈 Set your provider and API key (if needed), choose documents, then click **Apply & load documents**."
        )
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- `{s}`")

    if q := st.chat_input("Ask about revenue, risks, strategy…"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving context and generating answer…"):
                    result = agent.ask(q)
                answer = result.get("answer", "")
                st.markdown(answer or "_No answer._")
                sources = result.get("sources") or []
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- `{s}`")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as e:
                err = f"**Error:** {type(e).__name__} — {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})


if __name__ == "__main__":
    main()
