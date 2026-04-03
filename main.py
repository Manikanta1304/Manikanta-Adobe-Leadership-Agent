"""
AI Leadership Insight Agent — Entry Point

Usage examples:
  # Interactive mode
  python main.py

  # Single question
  python main.py --question "What is our current revenue trend?"

  # Re-index documents then answer
  python main.py --reload --question "Which departments are underperforming?"

  # Point to a custom documents directory
  python main.py --docs ./my_reports --question "What were Q3 key risks?"

  # Local free models (Ollama — no API key; install from https://ollama.com )
  #   ollama pull llama3.2
  #   python main.py --question "What is our revenue trend?"

  # Groq (free API tier — https://console.groq.com/ )
  #   setx GROQ_API_KEY "gsk_..."
  #   python main.py --question "What is our revenue trend?"
"""

import argparse
import json
import sys
import os

from agent import LeadershipInsightAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Leadership Insight Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--docs",
        default="./documents",
        help="Path to the folder containing company documents (default: ./documents)",
    )
    parser.add_argument(
        "--index",
        default="./vector_index",
        help="Path to store/load the FAISS vector index (default: ./vector_index)",
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="A single question to answer (omit for interactive mode)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Force re-ingestion and re-indexing of all documents",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of document chunks to retrieve per query (default: 6)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model id (overrides env; e.g. llama3.2, llama-3.3-70b-versatile for Groq)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output the result as JSON (useful for programmatic use)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    agent = LeadershipInsightAgent(
        docs_dir=args.docs,
        index_path=args.index,
        top_k=args.top_k,
        model=args.model,
    )
    if agent.llm_provider == "ollama":
        print(
            f"[Agent] Using local Ollama model '{agent.model}' "
            f"(start Ollama, then: ollama pull {agent.model})",
            file=sys.stderr,
        )
    elif agent.llm_provider == "groq":
        print(
            f"[Agent] Using Groq model '{agent.model}' "
            "(https://console.groq.com/docs/models)",
            file=sys.stderr,
        )

    # Single-question mode
    if args.question:
        agent.ingest_documents(force_rebuild=args.reload)
        result = agent.ask(args.question)

        if args.as_json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "═" * 60)
            print(f"Q: {result['question']}\n")
            print(result["answer"])
            print(f"\nSources: {', '.join(result['sources'])}")
            print("═" * 60)
    else:
        # Interactive mode
        agent.run_interactive()


if __name__ == "__main__":
    main()
