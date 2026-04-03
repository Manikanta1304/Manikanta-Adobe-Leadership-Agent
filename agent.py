"""
AI Leadership Insight Agent
Core RAG agent: retrieves relevant document chunks and generates
grounded answers via Claude, OpenAI, Groq (free tier), or local Ollama.
"""

import os
from typing import Literal, Optional

import anthropic
from openai import AuthenticationError, OpenAI

from ingestion import DocumentIngester
from vector_store import VectorStore


def _env_strip(name: str) -> str:
    """Environment value with leading/trailing whitespace removed (common copy/paste issue)."""
    return (os.environ.get(name) or "").strip()


def _strip_opt(value: Optional[str]) -> str:
    return (value or "").strip()


SYSTEM_PROMPT = """You are an elite executive intelligence analyst embedded within the organization. 
Your role is to provide leadership with precise, data-driven insights drawn exclusively from internal company documents.

Guidelines:
- Answer ONLY from the provided document context. Never speculate or use external knowledge.
- Be concise and direct — executives value brevity. Lead with the key finding.
- Cite the source document for every claim (e.g. [Q3 Report 2024]).
- If the documents don't contain enough information to answer, say so clearly.
- Structure multi-part answers with clear sections.
- Highlight risks, anomalies, or urgent items prominently.
- Use numbers and metrics whenever available.
"""


class LeadershipInsightAgent:
    """
    RAG-based agent that answers leadership questions
    grounded in company documents.
    """

    def __init__(
        self,
        docs_dir: str = "./documents",
        index_path: str = "./vector_index",
        model: Optional[str] = None,
        top_k: int = 6,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        *,
        llm_provider: Optional[Literal["anthropic", "openai", "groq", "ollama"]] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        groq_base_url: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_api_key: Optional[str] = None,
    ):
        self.docs_dir = docs_dir
        self.index_path = index_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if llm_provider is None:
            self.llm_provider = self._resolve_llm_provider()
        else:
            self.llm_provider = llm_provider

        a_key = _strip_opt(anthropic_api_key) or _env_strip("ANTHROPIC_API_KEY")
        o_key = _strip_opt(openai_api_key) or _env_strip("OPENAI_API_KEY")
        g_key = _strip_opt(groq_api_key) or _env_strip("GROQ_API_KEY")

        if llm_provider is not None:
            self._require_keys_for_provider(self.llm_provider, a_key, o_key, g_key)

        if self.llm_provider == "anthropic":
            self.model = model or "claude-sonnet-4-20250514"
            self._anthropic = anthropic.Anthropic(api_key=a_key)
            self._openai = None
        elif self.llm_provider == "openai":
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self._anthropic = None
            self._openai = OpenAI(api_key=o_key)
        elif self.llm_provider == "groq":
            self.model = model or os.environ.get(
                "GROQ_MODEL", "llama-3.3-70b-versatile"
            )
            self._anthropic = None
            groq_base = (
                _strip_opt(groq_base_url)
                or os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            ).rstrip("/")
            if not groq_base.endswith("/v1"):
                groq_base = f"{groq_base}/v1"
            self._openai = OpenAI(base_url=groq_base, api_key=g_key)
        else:
            self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
            self._anthropic = None
            base = (
                _strip_opt(ollama_base_url)
                or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            ).rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            o_llm_key = (
                _strip_opt(ollama_api_key)
                or _env_strip("OLLAMA_API_KEY")
                or "ollama"
            )
            self._openai = OpenAI(base_url=base, api_key=o_llm_key)

        self.ingester = DocumentIngester(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(index_path=index_path)

        self._index_built = False

    @staticmethod
    def _require_keys_for_provider(
        provider: Literal["anthropic", "openai", "groq", "ollama"],
        a_key: str,
        o_key: str,
        g_key: str,
    ) -> None:
        if provider == "anthropic" and not a_key:
            raise ValueError("Anthropic requires an API key.")
        if provider == "openai" and not o_key:
            raise ValueError("OpenAI requires an API key.")
        if provider == "groq" and not g_key:
            raise ValueError("Groq requires an API key.")

    @staticmethod
    def _resolve_llm_provider() -> Literal["anthropic", "openai", "groq", "ollama"]:
        override = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
        if override in ("anthropic", "openai", "groq", "ollama"):
            if override == "anthropic" and not _env_strip("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY."
                )
            if override == "openai" and not _env_strip("OPENAI_API_KEY"):
                raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY.")
            if override == "groq" and not _env_strip("GROQ_API_KEY"):
                raise RuntimeError("LLM_PROVIDER=groq requires GROQ_API_KEY.")
            return override  # type: ignore[return-value]

        has_anthropic = bool(_env_strip("ANTHROPIC_API_KEY"))
        has_openai = bool(_env_strip("OPENAI_API_KEY"))
        has_groq = bool(_env_strip("GROQ_API_KEY"))
        if has_anthropic and has_openai:
            return "anthropic"
        if has_anthropic:
            return "anthropic"
        if has_openai:
            return "openai"
        if has_groq:
            return "groq"
        return "ollama"

    def _complete(self, user_message: str) -> str:
        if self.llm_provider == "anthropic":
            assert self._anthropic is not None
            response = self._anthropic.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        assert self._openai is not None
        try:
            response = self._openai.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
        except AuthenticationError as err:
            if self.llm_provider == "groq":
                raise RuntimeError(
                    "Groq returned 401 (invalid API key). Open https://console.groq.com/keys , "
                    "create a new key (starts with gsk_), set GROQ_API_KEY in this shell, and retry. "
                    "Unset GROQ_API_KEY to fall back to local Ollama."
                ) from err
            if self.llm_provider == "openai":
                raise RuntimeError(
                    "OpenAI returned 401 (invalid API key). Check OPENAI_API_KEY."
                ) from err
            raise

        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    # ------------------------------------------------------------------ #
    #  Document Ingestion                                                  #
    # ------------------------------------------------------------------ #

    def ingest_documents(self, force_rebuild: bool = False) -> dict:
        """
        Load documents from docs_dir, chunk them, embed them,
        and persist the FAISS index.

        Returns a summary dict with stats.
        """
        if not force_rebuild and self.vector_store.load():
            print(f"[Agent] Loaded existing index from '{self.index_path}'.")
            self._index_built = True
            return {"status": "loaded_from_cache", "chunks": self.vector_store.size()}

        print(f"[Agent] Ingesting documents from '{self.docs_dir}' ...")
        chunks = self.ingester.ingest(self.docs_dir)

        if not chunks:
            raise ValueError(
                f"No documents found in '{self.docs_dir}'. "
                "Add PDF, DOCX, or TXT files and try again."
            )

        print(f"[Agent] Indexing {len(chunks)} chunks ...")
        self.vector_store.build(chunks)
        self.vector_store.save()

        self._index_built = True
        stats = {
            "status": "built",
            "documents": len({c["source"] for c in chunks}),
            "chunks": len(chunks),
        }
        print(f"[Agent] Index ready. {stats}")
        return stats

    # ------------------------------------------------------------------ #
    #  Querying                                                            #
    # ------------------------------------------------------------------ #

    def ask(self, question: str) -> dict:
        """
        Answer a leadership question using RAG.

        Returns:
            {
                "question": str,
                "answer": str,
                "sources": list[str],
                "context_chunks": int,
            }
        """
        if not self._index_built:
            self.ingest_documents()

        # 1. Retrieve relevant chunks
        retrieved = self.vector_store.search(question, k=self.top_k)
        if not retrieved:
            return {
                "question": question,
                "answer": "No relevant information found in the company documents.",
                "sources": [],
                "context_chunks": 0,
            }

        # 2. Build context block
        context_parts = []
        sources = []
        for chunk in retrieved:
            src = chunk["source"]
            if src not in sources:
                sources.append(src)
            context_parts.append(
                f"[Source: {src}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 3. Call LLM
        user_message = (
            f"DOCUMENT CONTEXT:\n\n{context}\n\n"
            f"LEADERSHIP QUESTION:\n{question}"
        )

        answer = self._complete(user_message)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context_chunks": len(retrieved),
        }

    # ------------------------------------------------------------------ #
    #  Interactive Session                                                 #
    # ------------------------------------------------------------------ #

    def run_interactive(self):
        """Start an interactive Q&A session in the terminal."""
        print("\n" + "═" * 60)
        print("  AI LEADERSHIP INSIGHT AGENT")
        print("  Type 'exit' to quit | 'reload' to re-index documents")
        print("═" * 60 + "\n")

        self.ingest_documents()

        while True:
            try:
                question = input("\n🔍 Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[Agent] Session ended.")
                break

            if not question:
                continue
            if question.lower() == "exit":
                print("[Agent] Goodbye.")
                break
            if question.lower() == "reload":
                self.ingest_documents(force_rebuild=True)
                continue

            result = self.ask(question)

            print("\n" + "─" * 60)
            print(f"📋 ANSWER\n")
            print(result["answer"])
            print(f"\n📂 Sources ({result['context_chunks']} chunks): {', '.join(result['sources'])}")
            print("─" * 60)
