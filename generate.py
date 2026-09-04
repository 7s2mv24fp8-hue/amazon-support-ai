"""
Generate text from a trained checkpoint.

Usage:
    python generate.py --prompt "Once upon a time"
    python generate.py --prompt "def add(a, b):" --tokens 300 --temperature 0.8
"""

import argparse

import torch

from model import GPT
from tokenizer import CharTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt_dir="checkpoints"):
    ckpt = torch.load(f"{ckpt_dir}/model.pt", map_location=DEVICE)
    tok = CharTokenizer.load(f"{ckpt_dir}/vocab.json")
    model = GPT(**ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    model, tok = load(args.ckpt_dir)

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long).to(DEVICE)
    out = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
