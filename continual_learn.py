"""
Continual learning loop: keeps the model "learning on its own" after
the initial training run, by periodically absorbing new text.

How it works:
  1. Drop .txt files into the `incoming/` folder whenever you want the
     model to learn something new (notes, articles, conversation logs,
     anything).
  2. Run this script. On each cycle it:
       - scans `incoming/` for files it hasn't processed yet
       - extends the vocab if new characters appear
       - fine-tunes the existing checkpoint on that new text for a
         small number of steps (not from scratch - it keeps what it
         already knows)
       - moves processed files into `incoming/processed/`
       - saves the updated checkpoint
  3. It sleeps, then repeats - so you can leave it running and just
     keep dropping new files in.

Why this stays "gist" learning and doesn't just memorize verbatim:
  - Each new file is trained for only a handful of steps at a low
    learning rate, on randomly sampled chunks (not the full text in
    order), which nudges the weights toward the statistical patterns
    in the text rather than exact sequences.
  - Steps and learning rate are kept intentionally low relative to
    file size - increase --steps_per_doc if you want it to absorb
    fewer, bigger documents harder, but going too high on a single
    small doc risks verbatim memorization (overfitting), not gist
    learning.

Usage:
    python continual_learn.py                 # one pass over incoming/, then exit
    python continual_learn.py --watch          # keep running, checking every 30s
    python continual_learn.py --steps_per_doc 40 --lr 1e-4
"""

import argparse
import os
import shutil
import time

import torch

from model import GPT
from tokenizer import CharTokenizer
from memory import MemoryBank

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(ckpt_dir):
    ckpt = torch.load(f"{ckpt_dir}/model.pt", map_location=DEVICE)
    tok = CharTokenizer.load(f"{ckpt_dir}/vocab.json")
    model = GPT(**ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    return model, tok, ckpt["config"]


def save_checkpoint(model, tok, config, ckpt_dir):
    torch.save({"model_state": model.state_dict(), "config": config}, f"{ckpt_dir}/model.pt")
    tok.save(f"{ckpt_dir}/vocab.json")


def finetune_on_text(model, tok, text, config, steps, lr, batch_size=16):
    if len(text) < config["block_size"] + 1:
        # too short for a full context window - pad by repeating
        text = (text * (config["block_size"] // max(len(text), 1) + 2))

    data = torch.tensor(tok.encode(text), dtype=torch.long).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    block_size = config["block_size"]
    for step in range(steps):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    return loss.item()


def process_new_files(model, tok, config, ckpt_dir, incoming_dir, steps_per_doc, lr,
                      memory: MemoryBank):
    processed_dir = os.path.join(incoming_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    files = [f for f in os.listdir(incoming_dir)
             if f.endswith(".txt") and os.path.isfile(os.path.join(incoming_dir, f))]

    if not files:
        return 0

    for fname in files:
        path = os.path.join(incoming_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            shutil.move(path, os.path.join(processed_dir, fname))
            continue

        # --- Memory dump: save a gist BEFORE fine-tuning so the info
        #     is never fully lost even if future training overwrites it.
        keywords = memory.dump(text, source=fname)
        print(f"  [memory] dumped '{fname}' under keywords: {keywords}")

        # No need to merge_vocab or resize when using tiktoken BPE tokenization.
        final_loss = finetune_on_text(model, tok, text, config, steps_per_doc, lr)
        print(f"  learned from {fname} ({len(text):,} chars), final loss {final_loss:.3f}")

        save_checkpoint(model, tok, config, ckpt_dir)
        shutil.move(path, os.path.join(processed_dir, fname))

    return len(files)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--incoming_dir", default="incoming")
    parser.add_argument("--memory_dir", default="memory_bank")
    parser.add_argument("--steps_per_doc", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--watch", action="store_true", help="keep running, polling for new files")
    parser.add_argument("--poll_seconds", type=int, default=30)
    args = parser.parse_args()

    model, tok, config = load_checkpoint(args.ckpt_dir)
    memory = MemoryBank(bank_dir=args.memory_dir)
    stats = memory.stats()
    print(f"Loaded checkpoint ({config['vocab_size']} vocab). Watching {args.incoming_dir}/ ...")
    print(f"Memory bank: {stats['keywords']} keywords, {stats['entries']} entries, "
          f"{stats['size_bytes']} bytes")

    while True:
        n = process_new_files(model, tok, config, args.ckpt_dir, args.incoming_dir,
                               args.steps_per_doc, args.lr, memory)
        if n:
            print(f"Processed {n} new file(s). Checkpoint updated.")
        elif not args.watch:
            print("No new files in incoming/.")

        if not args.watch:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
