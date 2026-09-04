"""
Train the GPT model from scratch on data/corpus.txt.

This does self-supervised next-token prediction: the model is shown a
chunk of text and has to predict, at every position, what character
comes next. There are no human-written labels - the "supervision"
comes entirely from the text itself. Over many steps this pushes the
model toward the GIST of the corpus (its vocabulary, grammar, style,
and recurring associations) rather than toward memorizing exact
strings, because it never sees the same exact chunk placement twice
and only has enough capacity to pick up general patterns.

Usage:
    python train.py                     # use defaults below
    python train.py --steps 3000        # train longer
    python train.py --data data/mine.txt
"""

import argparse
import os
import time

import torch

from model import GPT
from tokenizer import CharTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=50):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/corpus.txt")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=250)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters from {args.data}")

    tok = CharTokenizer(text=text)
    print(f"Vocab size: {tok.vocab_size}")

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters, training on {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, DEVICE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train loss {losses['train']:.3f} | "
                  f"val loss {losses['val']:.3f} | {elapsed:.1f}s")

    ckpt_path = os.path.join(args.out_dir, "model.pt")
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": tok.vocab_size,
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "dropout": args.dropout,
        },
    }, ckpt_path)
    tok.save(vocab_path)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved vocab to {vocab_path}")


if __name__ == "__main__":
    main()
