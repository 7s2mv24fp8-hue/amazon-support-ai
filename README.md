# 🛒 Amazon India Support AI — GPT-powered Customer Executive

A from-scratch GPT language model trained as an **Amazon India Customer Support Executive**, with a fully autonomous self-play training pipeline that generates and learns from conversations using a free local LLM (Ollama).

> Built with: PyTorch · tiktoken · Ollama (llama3.2:1b) · Groq API (optional)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Support Executive** | `support_chat.py` — "Priya" from Amazon India Support, answers 14 intent categories |
| 🧠 **From-scratch GPT** | Transformer model (same architecture as GPT-2/3/4) trained entirely from scratch |
| 🔄 **Continual Learning** | `continual_learn.py` — keeps learning from new `.txt` files dropped in `incoming/` |
| 🎭 **Self-Play Trainer** | `self_play_trainer.py` — uses Ollama/Groq to generate training conversations automatically |
| 📚 **Knowledge Base** | Amazon India official guidelines covering orders, payments, delivery, returns, EMI, UPI, Prime |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Self-Play Training Pipeline                 │
│                                                         │
│  llama3.2:1b (Ollama, free, local)                      │
│       │                                                 │
│       ├── plays "Customer"  → realistic Hindi/English   │
│       │                        questions & complaints   │
│       │                                                 │
│       └── plays "Agent"    → policy-accurate responses  │
│                ↓                                        │
│       incoming/amazon_conv_*.txt                        │
│                ↓                                        │
│       continual_learn.py → fine-tunes GPT checkpoint    │
│                ↓                                        │
│       checkpoints/model.pt  ←  gets smarter every loop  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Files

| File | Purpose |
|---|---|
| `model.py` | Transformer model: attention, feedforward blocks, embeddings |
| `tokenizer.py` | BPE tokenizer via tiktoken (GPT-2 vocabulary) |
| `train.py` | Initial training run on a corpus |
| `generate.py` | Generate text from a trained checkpoint |
| `continual_learn.py` | Absorbs new `.txt` files from `incoming/` into the model |
| `memory.py` | Keyword-indexed memory bank for document gists |
| `recall.py` | Query the memory bank |
| `support_chat.py` | **Amazon India Support chatbot** — interactive terminal UI |
| `self_play_trainer.py` | **Auto-generates training conversations** via Ollama/Groq/templates |
| `corpus.txt` | Sample training corpus |
| `amazon_support_knowledge.txt` | Amazon India policy knowledge base (Q&A format) |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the base model

```bash
python3 train.py --steps 2000
```

### 3. Chat with the Amazon Support Executive (rule-based, no model needed)

```bash
python3 support_chat.py
```

### 4. Chat with your trained GPT model as fallback

```bash
python3 support_chat.py --model checkpoints/
```

---

## 🎭 Self-Play Training Pipeline

Automatically generate Amazon support conversations and train the model on them — **completely free**.

### Option A — Ollama (free, local, best quality)

```bash
# Install Ollama from https://ollama.com
brew install ollama
brew services start ollama
ollama pull llama3.2:1b

# Generate 20 conversations and train — repeat every 90 seconds forever
python3 self_play_trainer.py \
  --mode ollama \
  --ollama_model llama3.2:1b \
  --num 20 \
  --watch \
  --train \
  --batch_interval 90
```

### Option B — Groq (free API, llama3-70b quality)

```bash
# Get free key at https://console.groq.com
export GROQ_API_KEY=your_key_here

python3 self_play_trainer.py \
  --mode groq \
  --num 50 \
  --watch \
  --train
```

### Option C — Template mode (zero dependencies)

```bash
# Works immediately, no LLM needed
python3 self_play_trainer.py --mode template --num 100
```

---

## 💬 Support Executive Demo

```
══════════════════════════════════════════════════════════════════════
  🛒  AMAZON INDIA  |  Customer Support Executive
  Powered by AI — Grounded in Amazon Official Guidelines
══════════════════════════════════════════════════════════════════════

🤖 Priya:
Hello! 👋 I'm Priya from Amazon India Customer Support.
How can I assist you today?

👤 You: my payment failed but money was deducted

🤖 Priya:
I completely understand how frustrating it is when a payment fails.
...
To revise payment for a pending order:
1. Go to Your Orders on Amazon India
2. Find the order with payment pending
3. Click "Revise Payment" and complete using a different method
...
```

---

## 🧠 Supported Intent Categories (14 topics)

| Intent | Example Queries |
|---|---|
| Cannot Place Order | "order not going through", "buy now not working" |
| Payment Failed | "payment declined", "money deducted order not confirmed" |
| UPI Failed | "gpay transaction failed", "upi timed out" |
| Track Order | "where is my package", "how to track order" |
| Late Delivery | "delivery delayed", "package stuck at hub" |
| Delivered Not Received | "shows delivered but not received", "missing package" |
| Cancel Order | "how to cancel", "cancel after shipment" |
| Returns & Refund | "how to return", "when will I get refund" |
| Damaged/Defective | "received broken item", "wrong product delivered" |
| Modify Order | "change quantity", "edit order details" |
| Undeliverable | "order marked undeliverable", "missed delivery" |
| EMI | "no cost EMI", "debit card EMI eligibility" |
| Unknown Charge | "unauthorized transaction", "unexpected deduction" |
| Prime Membership | "cancel prime", "stop auto renewal" |

---

## ⚙️ Training Configuration

| Parameter | Default | Description |
|---|---|---|
| `--steps` | 2000 | Initial training steps |
| `--n_layer` | 8 | Transformer layers |
| `--n_head` | 8 | Attention heads |
| `--n_embd` | 512 | Embedding dimension |
| `--block_size` | 256 | Context window size |
| `--lr` | 3e-4 | Learning rate |
| `--steps_per_doc` | 30 | Fine-tuning steps per new document |

---

## 📊 Self-Play Pipeline Parameters

| Parameter | Default | Description |
|---|---|---|
| `--num` | 20 | Conversations per batch |
| `--mode` | auto | `ollama` / `groq` / `template` |
| `--watch` | — | Run continuously in a loop |
| `--train` | — | Auto-run `continual_learn.py` after each batch |
| `--batch_interval` | 60 | Seconds between batches |
| `--turns` | random 2-5 | Conversation turns per file |

---

## 🔬 How Gist Learning Works

The model learns statistical **patterns**, not verbatim text:

- **Objective**: next-token prediction over randomly sampled chunks — never the full text in order
- **Capacity limit**: fewer parameters than corpus characters → forced to compress into patterns
- **Continual learning**: light fine-tuning (few steps, low LR) per document → avoids overfitting

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
