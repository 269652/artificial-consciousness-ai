import os
import sys
import argparse

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.memory.graph.controller import MemoryGraphController


def main():
    parser = argparse.ArgumentParser(description="Seed MemoryGraph with a fake conversation JSON")
    parser.add_argument("--file", "-f", default="data/conversations/sample_conversation.json", help="Path to conversation JSON")
    parser.add_argument("--user-tag", default="user_msg", help="Tag for user messages")
    parser.add_argument("--assistant-tag", default="assistant_msg", help="Tag for assistant messages")
    args = parser.parse_args()

    controller = MemoryGraphController()
    summary = controller.seed_from_conversation_json(args.file, tag_user=args.user_tag, tag_assistant=args.assistant_tag)

    print("Seed complete:")
    print(summary)

if __name__ == "__main__":
    main()
