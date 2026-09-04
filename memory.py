"""
memory.py — Lightweight memory bank for the GPT model.

The model can "dump" a compressed gist of any text it is about to learn,
keyed by auto-extracted keywords. Later, any word/phrase can be used to
retrieve the stored gist back — so nothing is truly forgotten.

Storage format: a single JSON file (`memory_bank/index.json`)
Each entry:  { "keyword": [ {"gist": "...", "source": "...", "ts": ...} ] }

Memory is intentionally cheap:
  - Each gist is capped at MAX_GIST_CHARS characters (default 300).
  - Keywords are extracted by token frequency (top-N words).
  - The full index is a plain JSON file — no vector DB, no embeddings.
"""

import json
import os
import re
import time
from collections import Counter


MAX_GIST_CHARS = 300   # characters stored per memory entry
TOP_K_KEYWORDS = 8     # how many keywords to index per document
STOPWORDS = set(
    "the a an and or but in on at to of for with is are was were be been "
    "being have has had do does did will would could should may might "
    "this that these those it its i you he she we they me him her us "
    "them my your his our their s t re ll ve d m just not no by from "
    "as so if then than when where who what which how all also both "
    "each more most other some such up out into over after".split()
)


class MemoryBank:
    def __init__(self, bank_dir="memory_bank"):
        self.bank_dir = bank_dir
        self.index_path = os.path.join(bank_dir, "index.json")
        os.makedirs(bank_dir, exist_ok=True)
        self.db = self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.db, f, ensure_ascii=False, separators=(",", ":"))

    def _extract_keywords(self, text):
        """Return the top-K meaningful words from text."""
        words = re.findall(r"[a-z]+", text.lower())
        words = [w for w in words if len(w) > 3 and w not in STOPWORDS]
        freq = Counter(words)
        return [w for w, _ in freq.most_common(TOP_K_KEYWORDS)]

    def _compress(self, text):
        """Trim text to a compact gist."""
        text = text.strip()
        if len(text) <= MAX_GIST_CHARS:
            return text
        half = MAX_GIST_CHARS // 2
        return text[:half] + " ... " + text[-half:]

    def dump(self, text, source="unknown"):
        """
        Store a gist of `text` into the memory bank, indexed by
        auto-extracted keywords. Returns the list of keywords used.
        """
        gist = self._compress(text)
        keywords = self._extract_keywords(text)
        entry = {"gist": gist, "source": source, "ts": int(time.time())}

        for kw in keywords:
            self.db.setdefault(kw, [])
            # Avoid duplicate gists for the same source
            if not any(e["source"] == source for e in self.db[kw]):
                self.db[kw].append(entry)

        self._save()
        return keywords

    def recall(self, trigger, top_n=3):
        """
        Return up to `top_n` gist entries that match `trigger`.
        Trigger can be a single word or a phrase — any matching keyword wins.
        """
        trigger_words = re.findall(r"[a-z]+", trigger.lower())
        hits = {}  # source -> entry (deduplicate by source)

        for word in trigger_words:
            for kw, entries in self.db.items():
                if word in kw or kw in word:
                    for entry in entries:
                        hits[entry["source"]] = entry

        results = sorted(hits.values(), key=lambda e: e["ts"], reverse=True)
        return results[:top_n]

    def recall_as_text(self, trigger, top_n=3):
        """
        Convenience wrapper — returns recalled memories as a single
        string ready to prepend to a generation prompt.
        """
        entries = self.recall(trigger, top_n=top_n)
        if not entries:
            return ""
        parts = [f"[MEMORY from '{e['source']}']: {e['gist']}" for e in entries]
        return "\n".join(parts)

    def stats(self):
        n_keys = len(self.db)
        n_entries = sum(len(v) for v in self.db.values())
        size_bytes = os.path.getsize(self.index_path) if os.path.exists(self.index_path) else 0
        return {"keywords": n_keys, "entries": n_entries, "size_bytes": size_bytes}
