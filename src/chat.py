"""CLI chat entrypoint.
Usage: python -m src.chat --session mycampaign
"""
import argparse
from rich.console import Console
from rich.markdown import Markdown
from .agent import RagAgent

console = Console()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="default", help="Session / campaign name for memory persistence")
    return ap.parse_args()

def main():
    args = parse_args()
    agent = RagAgent(session=args.session)
    console.print(f"[bold green]AI DnD RAG Agent - session '{args.session}'[/bold green]")
    console.print("Type your questions. Prefix with 'Remember this:' to store a fact. Type 'exit' to quit.\n")
    while True:
        try:
            user_input = console.input("[bold cyan]You> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            console.print("Goodbye.")
            break
        with console.status("Thinking..."):
            answer = agent.answer(user_input)
        console.print(Markdown(answer))

if __name__ == "__main__":
    main()
