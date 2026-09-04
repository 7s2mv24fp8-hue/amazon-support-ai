"""
recall.py — Recall from the memory bank by any trigger word/phrase.

Usage:
    python recall.py "lighthouse"
    python recall.py "ocean storm"           # phrase — any matching keyword wins
    python recall.py --top 5 "science"       # return up to 5 memories
    python recall.py --stats                 # show memory bank statistics
"""

import argparse
from memory import MemoryBank


def main():
    parser = argparse.ArgumentParser(description="Recall from the model's memory bank.")
    parser.add_argument("trigger", nargs="?", default="", help="Word or phrase to recall")
    parser.add_argument("--top", type=int, default=3, help="Max memories to show")
    parser.add_argument("--memory_dir", default="memory_bank")
    parser.add_argument("--stats", action="store_true", help="Show memory bank stats and exit")
    args = parser.parse_args()

    memory = MemoryBank(bank_dir=args.memory_dir)

    if args.stats:
        s = memory.stats()
        print(f"Memory bank stats:")
        print(f"  Keywords indexed : {s['keywords']}")
        print(f"  Total entries    : {s['entries']}")
        print(f"  Disk usage       : {s['size_bytes']} bytes")
        return

    if not args.trigger:
        parser.print_help()
        return

    results = memory.recall(args.trigger, top_n=args.top)
    if not results:
        print(f"No memories found for trigger: '{args.trigger}'")
        return

    print(f"Found {len(results)} memory/memories for '{args.trigger}':\n")
    for i, entry in enumerate(results, 1):
        print(f"--- Memory {i} (from: {entry['source']}) ---")
        print(entry["gist"])
        print()


if __name__ == "__main__":
    main()
