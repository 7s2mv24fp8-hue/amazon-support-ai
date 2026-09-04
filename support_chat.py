"""
Amazon India Customer Support Executive
========================================

A multi-layered AI support chatbot grounded in Amazon India's official help policies.

Architecture:
  1. Intent Classifier: High-confidence regex pattern matching for fast, accurate policy answers
  2. RAG Knowledge Retriever: Semantic keyword scoring over amazon_support_knowledge.txt
  3. Ollama LLM Engine: Uses local llama3.2:1b with Amazon India Support persona & RAG context
  4. PyTorch Checkpoint Fallback: Local GPT model support
  5. Structured Helpful Fallback: Clear self-service options & Amazon India helpline (1800-1200-1637)
"""

import os
import re
import sys
import json
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional, Dict, List, Any

# ──────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE (Amazon India Official Policies & Guidelines)
# ──────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "cannot_place_order": {
        "keywords": ["place order", "cant order", "cannot order", "unable to order", "order failed", "unable to place", "order not placing"],
        "patterns": [
            r"(can'?t|cannot|unable|not able|problem|issue|trouble).*(place|make|do|complete|submit).*(order|purchase|buy)",
            r"(place|placing|make|making).*(order|purchase).*(problem|issue|fail|error|decline)",
            r"why (am i|can'?t i|can i not).*(order|buy|purchase)",
            r"order.*(not going through|failing|failed|error|stuck)",
        ],
        "response": """I understand you're having trouble placing an order. Here are the most common reasons and solutions:

1. 💳 **Payment Decline** — Your bank may have declined the transaction. Try another card, UPI, or Net Banking.
2. 📦 **Item Out of Stock** — The product may have just sold out.
3. 🌐 **Technical/Connectivity Issue** — Clear your app/browser cache and refresh.
4. 📍 **Undeliverable Pincode** — The seller might not currently deliver to your address.
5. 🔢 **Quantity Limit** — Certain high-demand items have a per-customer limit.
6. ⏳ **FBA + Seller Items Combined** — Mixed carts require extra processing time.

👉 **Quick Fix:** Check your pincode on the product page and verify your payment details in [Your Account](https://www.amazon.in/gp/css/your-orders-access).

Would you like help with a specific payment method or delivery address?"""
    },

    "modify_order": {
        "keywords": ["modify order", "change item", "change order", "edit order", "update order", "add item"],
        "patterns": [
            r"\b(change|modify|update|alter)\b.*\b(order|item|quantity|number|cart)\b",
            r"\b(order)\b.*\b(change|modify|update)\b",
            r"\b(add|remove)\b.*\b(item|product)\b.*\b(order)\b",
        ],
        "response": """Once an order is placed on Amazon India, **you cannot change or modify items or quantities**.

Here are your best options:
- ✅ **Cancel the order** and place a fresh order with your updated items.
- ✅ **Change shipping address/preferences** in [Your Orders](https://www.amazon.in/gp/css/your-orders-access) if the order has not yet entered the shipping process.

👉 **To cancel & re-order:** Go to **Your Orders → Select Item → Cancel Items**."""
    },

    "cancel_order": {
        "keywords": ["cancel order", "cancel item", "cancellation", "how to cancel", "want to cancel", "stop order"],
        "patterns": [
            r"cancel.*(order|item|purchase|package|delivery)",
            r"(order|item|package).*(cancel|cancellation)",
            r"how (do i|to|can i).*(cancel)",
            r"want to cancel",
        ],
        "response": """Here is how to cancel an order on Amazon India:

**Before Shipment:**
1. Go to [Your Orders](https://www.amazon.in/gp/css/your-orders-access).
2. Select the order and click **"Cancel Items"**.
3. Select your cancellation reason (optional) and confirm.

**After Shipment:**
1. Go to [Your Orders](https://www.amazon.in/gp/css/your-orders-access).
2. Click **"Request Cancellation"** or simply refuse the delivery at your doorstep.
3. Once returned to Amazon, your full refund will be initiated automatically.

💡 Refunds to original payment methods (UPI/Cards) take **2–5 business days**, while Amazon Pay balance is **instant**."""
    },

    "track_order": {
        "keywords": ["track order", "track package", "where is my order", "order status", "shipment tracking", "delivery status"],
        "patterns": [
            r"track.*(order|package|shipment|delivery|item)",
            r"(order|package|shipment|item).*(track|status|where|location)",
            r"where is my (order|package|parcel|shipment|item)",
            r"(status|update).*(order|delivery|shipment)",
        ],
        "response": """To track your package in real-time:

1. Go to [Your Orders](https://www.amazon.in/gp/css/your-orders-access).
2. Find your order and click **"Track Package"**.
3. Click **"See all updates"** to view live location scans and estimated delivery time.
4. You can also view the **Courier Partner** (e.g. Amazon Logistics, Delhivery, BlueDart) and Tracking ID.

Is your delivery overdue or would you like help checking its status?"""
    },

    "late_delivery": {
        "keywords": ["late delivery", "delayed", "delay", "order late", "package not arrived", "has not arrived", "not received yet"],
        "patterns": [
            r"(late|delayed|delay|slow|overdue).*(delivery|order|shipment|package)",
            r"(delivery|order|shipment).*(late|delayed|delay|not arrived|hasn'?t arrived|overdue)",
            r"(expected|estimated).*(delivery|date).*(passed|gone|over)",
            r"not (received|delivered|arrived).*(yet|still)",
        ],
        "response": """I apologize for the delay with your order! Common reasons include transit delays, weather conditions, or local carrier constraints.

**What you can do right now:**
1. Check updated delivery estimates under [Your Orders](https://www.amazon.in/gp/css/your-orders-access) → **Track Package**.
2. Verify your delivery address and contact number are accurate.
3. Most delayed packages arrive within **24–48 hours** of the original date.

🛡️ If your item was sold by a third-party seller, you are fully covered under **Amazon's A-to-Z Guarantee**."""
    },

    "delivered_not_received": {
        "keywords": ["shows delivered", "marked delivered", "delivered but not received", "haven't received", "missing package"],
        "patterns": [
            r"(shows|marked|status).*(delivered|delivery).*(but|however|yet|still).*(not|haven'?t|didn'?t).*(received|got|arrived)",
            r"(not|haven'?t|didn'?t).*(received|got).*(shows|marked|status).*(delivered)",
            r"says delivered but (i haven'?t|not) (received|got)",
            r"(package|order|parcel).*(missing|lost).*(delivered|delivery)",
            r"tracking shows delivered",
        ],
        "response": """If tracking shows "Delivered" but you don't have your package, please follow these steps:

1. **Check Delivery Location & Neighbors:** Check with family members, security guard, receptionist, or neighbors who might have accepted on your behalf.
2. **Check Message Center:** Look for delivery photo / OTP verification messages in your Amazon app.
3. **Wait 24 Hours:** In rare cases, delivery associates mark a package delivered slightly before arrival.
4. **Contact Support:** If you still cannot locate it after 24 hours, contact customer support or file an **A-to-Z Guarantee Claim** in Your Orders."""
    },

    "damaged_defective": {
        "keywords": ["damaged", "defective", "broken", "wrong item", "fake", "counterfeit", "expired", "missing parts", "replacement"],
        "patterns": [
            r"(damaged|defective|broken|not working|faulty|defect|scratched|dented|torn|missing parts|wrong item|fake|counterfeit|expired|leaking|dead on arrival)",
            r"(received|got|delivered).*(wrong|damaged|broken|defective|different|fake|expired)",
            r"(product|item|order).*(damage|defect|issue|problem|wrong|broken|fake)",
            r"item.*(not matching|doesn'?t match|different|missing)",
        ],
        "response": """I'm very sorry you received a damaged or incorrect product! Amazon India provides a hassle-free replacement or refund:

**How to request a replacement / refund:**
1. Go to [Your Orders](https://www.amazon.in/gp/css/your-orders-access).
2. Click **"Return or Replace Items"** next to the affected product.
3. Select the reason (e.g. *Damaged*, *Defective*, or *Wrong item sent*).
4. Choose whether you'd prefer a **free replacement** or **full refund**.
5. Schedule a doorstep pickup date.

All products sold by Amazon or third-party sellers are covered by Amazon's return policies and A-to-Z Guarantee."""
    },

    "returns_refund": {
        "keywords": ["return", "refund", "how to return", "refund timeline", "money back", "return item", "refund status"],
        "patterns": [
            r"(return|refund|replace|replacement|exchange).*(order|product|item|package)",
            r"(order|product|item|package).*(return|refund|replace|replacement|exchange)",
            r"how (do i|to|can i).*(return|refund|replace|exchange)",
            r"(want to|need to).*(return|refund|get refund)",
            r"refund (status|when|timeline|time|process)",
        ],
        "response": """Here is the complete Return & Refund policy for Amazon India:

**To initiate a return:**
1. Go to [Your Orders](https://www.amazon.in/gp/css/your-orders-access) → Select item → **"Return or Replace Items"**.
2. Select your pickup address and schedule a convenient slot.

**Refund Timelines (after pickup/processing):**
| Payment Method | Refund Duration |
|---|---|
| **Amazon Pay Balance** | Instant (within 2 hours) |
| **UPI** | 2–4 business days |
| **Credit / Debit Cards** | 3–5 business days |
| **Net Banking** | 3–5 business days |
| **Cash on Delivery (COD)** | 2–4 days to linked Bank Account |

You can check your refund status anytime in **Your Orders**."""
    },

    "payment_failed": {
        "keywords": ["payment failed", "payment decline", "transaction failed", "payment error", "money deducted order not placed"],
        "patterns": [
            r"(payment|transaction).*(fail|failed|failing|decline|declined|not (going through|processed|working))",
            r"(fail|failed|error|decline|declined).*(payment|transaction|order)",
            r"(couldn'?t|could not|unable to).*(pay|make payment|complete payment|process payment)",
            r"payment (issue|problem|error)",
            r"(money|amount).*(deducted|debited).*(order not|no order)",
        ],
        "response": """Here's what to do if your payment failed or was declined:

**If money was deducted from your bank:**
- Don't worry! Your money is completely safe.
- If the order wasn't created, your bank will auto-refund the full amount within **3–5 business days**.

**To fix a pending / failed payment:**
1. Go to [Your Orders](https://www.amazon.in/gp/css/your-orders-access).
2. Click **"Revise Payment"** next to the pending order.
3. Select an alternate payment method (UPI, different card, or Net Banking)."""
    },

    "upi_failed": {
        "keywords": ["upi failed", "upi error", "gpay", "phonepe", "paytm", "bhim", "upi limit"],
        "patterns": [
            r"upi.*(fail|failed|not working|error|decline|declined|issue|timeout)",
            r"(fail|failed|error|issue).*(upi)",
            r"(gpay|phonepe|paytm|bhim).*(fail|issue|error)",
        ],
        "response": """UPI payment issues on Amazon India:

**Common causes & limits:**
- **App only:** Amazon UPI is supported only on the Amazon mobile app (not web/mobile browser).
- **Daily limit:** Standard bank limit is ₹1,00,000 per day.
- **New UPI account limit:** Up to ₹5,000 in the first 24 hours.
- **Bank server downtime:** Temporary bank connection timeouts.

**Recommended steps:**
1. Wait 15–20 minutes and check [Your Orders](https://www.amazon.in/gp/css/your-orders-access) → **Revise Payment**.
2. Or use Amazon Pay Balance, Credit/Debit Card, or Net Banking to complete checkout instantly."""
    },

    "unknown_charge": {
        "keywords": ["unknown charge", "unauthorized charge", "charged without order", "unrecognized payment", "charged twice"],
        "patterns": [
            r"(unknown|unauthorized|unexpected|strange|weird|unrecognized|duplicate|twice).*(charge|deduction|transaction|payment|debited)",
            r"(charge|deduction|transaction|payment).*(unknown|unauthorized|unexpected|strange|unrecognized|twice)",
            r"why (was i|am i).*(charged|debited)",
        ],
        "response": """If you see an unfamiliar or unexpected charge from Amazon:

1. **Check Prime Membership:** Verify if your annual/monthly Prime subscription renewed automatically under **Your Account → Prime**.
2. **Check Family Members:** Check if someone with shared access placed an order.
3. **Check Authorizations:** Bank temporary pre-authorizations for cancelled orders disappear within 48 hours.
4. **Report Unauthorized Charge:** If you suspect unauthorized access, contact Amazon Support immediately at **1800-1200-1637**."""
    },

    "prime": {
        "keywords": ["prime", "prime membership", "prime video", "cancel prime", "prime renewal", "prime cost"],
        "patterns": [
            r"(prime|prime membership|prime subscription)",
            r"(cancel|manage|renew|renewal).*(prime|membership|subscription)",
            r"prime.*(benefits|cancel|renew|cost|price|charged)",
        ],
        "response": """Amazon Prime India Overview & Management:

**Prime Benefits:**
- ⚡ Free 1-Day & 2-Day Delivery
- 🎬 Prime Video & Prime Music streaming
- 🏷️ Exclusive Deals & Early Access during sales

**Manage / Cancel Subscription:**
- Visit **Your Account → Manage Your Prime Membership**.
- You can turn off auto-renewal, switch plans (Monthly/Annual), or cancel anytime with a pro-rated refund."""
    },

    "contact_human": {
        "keywords": ["human agent", "talk to agent", "customer care number", "call amazon", "speak to someone", "support number", "helpline"],
        "patterns": [
            r"(human|person|agent|executive|representative|speak to|talk to|call).*(customer care|support|human|amazon|help)",
            r"(customer care|helpline|phone number|toll free|contact number).*(amazon)?",
            r"how (can|do) i (talk|speak|call) (to|with)",
        ],
        "response": """You can connect with Amazon India customer service via:

📞 **Toll-Free Helpline:** `1800-1200-1637` or `1800-3000-9009` (24x7)
💬 **Call-Me-Back Request:** Go to **Amazon App → Menu → Customer Service → Contact Us → Request a Call** (an executive will call you within 2 minutes).
📧 **Online Help Center:** [Amazon India Customer Service](https://www.amazon.in/gp/help/customer/display.html)"""
    },

    "greeting": {
        "keywords": ["hi", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening"],
        "patterns": [
            r"^(hi|hello|hey|good morning|good afternoon|good evening|namaste|hii|helo|helloo|sup|yo)[\s!.,]*$",
            r"^(hi|hello|hey).*(there|amazon|support|help)[\s!.,]*$",
        ],
        "response": None
    },

    "goodbye": {
        "keywords": ["bye", "goodbye", "thanks bye", "resolved", "sorted", "no further questions"],
        "patterns": [
            r"^(bye|goodbye|see you|thanks bye|thank you bye|that'?s all|nothing else|no that'?s all|all good now|resolved|sorted)[\s!.,]*$",
            r"(thank you|thanks).*(that'?s all|nothing else|goodbye|bye|that will be all)",
        ],
        "response": None
    },

    "thanks": {
        "keywords": ["thank you", "thanks", "tysm", "thx", "appreciate it"],
        "patterns": [
            r"^(thank you|thanks|thank u|thx|ty|tysm|thank you so much|many thanks)[\s!.,]*$",
            r"(thank you|thanks).*(help|assistance|support|info|information)",
        ],
        "response": None
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# RAG KNOWLEDGE RETRIEVER
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeRetriever:
    """Retrieves relevant chunks from amazon_support_knowledge.txt using keyword relevance."""

    def __init__(self, knowledge_path: Optional[str] = None):
        self.chunks: List[Dict[str, str]] = []
        if knowledge_path is None:
            default_path = os.path.join(os.path.dirname(__file__), "amazon_support_knowledge.txt")
            if os.path.exists(default_path):
                knowledge_path = default_path

        if knowledge_path and os.path.exists(knowledge_path):
            self._load_knowledge(knowledge_path)

    def _load_knowledge(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split by section headers or Q&A blocks
            sections = content.split("===")
            for section in sections:
                text = section.strip()
                if not text:
                    continue
                # Split further by Q: / A: pairs if present
                qa_pairs = re.split(r"(?=Q:)", text)
                for qa in qa_pairs:
                    qa_clean = qa.strip()
                    if qa_clean:
                        self.chunks.append({"text": qa_clean})
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        if not self.chunks:
            return []

        q_words = set(re.findall(r"\w+", query.lower()))
        if not q_words:
            return []

        scored = []
        for chunk in self.chunks:
            chunk_words = set(re.findall(r"\w+", chunk["text"].lower()))
            overlap = len(q_words & chunk_words)
            if overlap > 0:
                score = overlap / (len(q_words) ** 0.5)
                scored.append((score, chunk["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


# ──────────────────────────────────────────────────────────────────────────────
# OLLAMA GENERATION CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class OllamaGenerator:
    """Invokes local Ollama LLM (e.g. llama3.2:1b) for grounded customer responses."""

    def __init__(self, model_name: str = "llama3.2:1b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return any(self.model_name.split(":")[0] in m for m in models)
        except Exception:
            return False
        return False

    def generate_response(self, user_query: str, retrieved_context: str, history: List[Dict[str, str]]) -> Optional[str]:
        try:
            system_prompt = (
                "You are Priya, a polite, helpful, and professional Customer Support Executive from Amazon India. "
                "Your job is to assist Amazon India customers accurately based on Amazon India policies. "
                "Guidelines:\n"
                "- Greet politely and be empathetic.\n"
                "- Provide clear, step-by-step guidance.\n"
                "- Ground your answers strictly in Amazon India guidelines.\n"
                "- Keep responses concise and structured with bullet points or numbered steps where appropriate.\n"
                "- Never make up false policies."
            )

            prompt_content = f"Official Amazon India Policy Context:\n{retrieved_context}\n\nCustomer Inquiry: {user_query}"

            # Format conversation for Ollama chat API
            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-3:]:
                role = "assistant" if h.get("role") == "agent" else "user"
                messages.append({"role": role, "content": h.get("text", "")})
            messages.append({"role": "user", "content": prompt_content})

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 180
                }
            }

            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=25) as resp:
                if resp.status == 200:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    message = res_json.get("message", {}).get("content", "").strip()
                    if message:
                        return message
        except Exception as e:
            if self.host:
                pass
        return None


# ──────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFIER WITH FUZZY KEYWORD SCORING
# ──────────────────────────────────────────────────────────────────────────────

def classify_intent(user_input: str) -> str:
    """Match user input to an intent using regex patterns and keyword heuristics."""
    text = user_input.lower().strip()

    # 1. Regex Pattern Matching
    for intent, data in KNOWLEDGE_BASE.items():
        for pattern in data.get("patterns", []):
            if re.search(pattern, text):
                return intent

    # 2. Keyword Overlap Scoring
    tokens = set(re.findall(r"\w+", text))
    best_intent = "unknown"
    best_score = 0

    for intent, data in KNOWLEDGE_BASE.items():
        for kw in data.get("keywords", []):
            kw_tokens = set(re.findall(r"\w+", kw.lower()))
            if kw_tokens.issubset(tokens):
                score = len(kw_tokens) * 2
                if score > best_score:
                    best_score = score
                    best_intent = intent

    return best_intent


# ──────────────────────────────────────────────────────────────────────────────
# MAIN SUPPORT EXECUTIVE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class SupportExecutive:
    """Amazon India Customer Support Executive."""

    AGENT_NAME = "Priya"
    BRAND = "Amazon India"

    def __init__(self, model=None, tokenizer=None, config=None, debug=False, knowledge_path=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.debug = debug
        self.conversation_history: List[Dict[str, str]] = []
        self.customer_name: Optional[str] = None
        self.session_start = datetime.now()
        self.turn_count = 0

        self.retriever = KnowledgeRetriever(knowledge_path)
        self.ollama = OllamaGenerator()

    def _greeting(self) -> str:
        name_part = f", {self.customer_name}" if self.customer_name else ""
        return (
            f"Hello{name_part}! 👋 I'm **{self.AGENT_NAME}** from {self.BRAND} Customer Support.\n\n"
            f"I can assist you with your orders, tracking, refunds, payment issues, returns, and more.\n\n"
            f"How can I help you today?"
        )

    def _farewell(self) -> str:
        name_part = f", {self.customer_name}" if self.customer_name else ""
        return (
            f"Thank you for contacting {self.BRAND} Support{name_part}! 😊\n\n"
            f"I'm glad I could assist you today. Have a wonderful day and happy shopping! 🛒\n\n"
            f"*Session ended. Feel free to type any message to start a new chat.*"
        )

    def _thanks(self) -> str:
        return (
            f"You're very welcome! 😊 I'm always happy to help.\n\n"
            f"Is there anything else regarding your Amazon India orders or account I can assist you with?"
        )

    def _unknown_fallback(self, user_input: str) -> str:
        # Retrieve context from official knowledge base
        retrieved = self.retriever.retrieve(user_input, top_k=2)
        retrieved_text = "\n\n".join(retrieved) if retrieved else ""

        # 1. Try Ollama LLM if available
        if self.ollama.is_available():
            llm_reply = self.ollama.generate_response(user_input, retrieved_text, self.conversation_history)
            if llm_reply:
                return llm_reply

        # 2. Try RAG direct snippet if score is good
        if retrieved_text and len(retrieved_text) > 40:
            return f"Based on Amazon India's official guidelines:\n\n{retrieved_text}\n\n*Does this answer your question, or would you like to speak to an executive?*"

        # 3. Try PyTorch model fallback
        model_reply = self._try_pytorch_generate(user_input)
        if model_reply:
            return model_reply

        # 4. Comprehensive Structured Fallback
        return (
            f"I want to make sure you get the exact help you need! Here are quick actions for common queries:\n\n"
            f"  📦 **Track or Cancel Order:** [Your Orders](https://www.amazon.in/gp/css/your-orders-access)\n"
            f"  💳 **Payment Declined / Retry:** [Revise Payment](https://www.amazon.in/gp/css/your-orders-access)\n"
            f"  🔄 **Return & Replacement:** [Return Center](https://www.amazon.in/returns)\n"
            f"  ⭐ **Prime Membership:** [Manage Prime](https://www.amazon.in/prime)\n"
            f"  📞 **Speak with Support Executive:** Call toll-free **1800-1200-1637** (24x7)\n\n"
            f"Could you please describe your issue with a bit more detail (e.g., order status, return, refund) so I can guide you?"
        )

    def _try_pytorch_generate(self, prompt: str) -> Optional[str]:
        if self.model is None or self.tokenizer is None:
            return None
        try:
            import torch
            context = f"Customer: {prompt}\nSupport Agent Priya:"
            ids = self.tokenizer.encode(context)
            x = torch.tensor([ids], dtype=torch.long)
            device = next(self.model.parameters()).device
            x = x.to(device)
            self.model.eval()
            with torch.no_grad():
                block_size = self.config.get("block_size", 128)
                for _ in range(80):
                    x_cond = x[:, -block_size:]
                    logits, _ = self.model(x_cond)
                    probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    x = torch.cat([x, next_id], dim=1)
                    decoded = self.tokenizer.decode([next_id.item()])
                    if decoded in ["\n\n", "Customer:"]:
                        break
            generated = self.tokenizer.decode(x[0].tolist()[len(ids):]).split("\n\n")[0].strip()
            if len(generated) > 25:
                return generated
        except Exception:
            pass
        return None

    def respond(self, user_input: str) -> str:
        """Process input and return a grounded, structured support response."""
        clean_input = user_input.strip()
        if not clean_input:
            return "Please type a message so I can assist you."

        self.turn_count += 1
        self.conversation_history.append({"role": "customer", "text": clean_input})

        # Name extraction
        name_match = re.search(r"(?:i(?:'m| am)|my name is|this is|call me)\s+([A-Z][a-z]+)", clean_input, re.IGNORECASE)
        if name_match and not self.customer_name:
            self.customer_name = name_match.group(1).capitalize()

        intent = classify_intent(clean_input)

        if self.debug:
            print(f"  [DEBUG] Matched Intent: {intent}")

        # Intent Routing
        if intent == "greeting":
            response = self._greeting()
        elif intent == "goodbye":
            response = self._farewell()
        elif intent == "thanks":
            response = self._thanks()
        elif intent != "unknown" and intent in KNOWLEDGE_BASE and KNOWLEDGE_BASE[intent].get("response"):
            response = KNOWLEDGE_BASE[intent]["response"]
        else:
            response = self._unknown_fallback(clean_input)

        self.conversation_history.append({"role": "agent", "text": response})
        return response


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADER
# ──────────────────────────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_dir: str):
    """Load GPT model and tokenizer from checkpoint directory if available."""
    try:
        import torch
        from model import GPT
        from tokenizer import CharTokenizer

        model_path = os.path.join(ckpt_dir, "model.pt")
        vocab_path = os.path.join(ckpt_dir, "vocab.json")

        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            return None, None, None

        ckpt = torch.load(model_path, map_location="cpu")
        tok = CharTokenizer.load(vocab_path)
        model = GPT(**ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, tok, ckpt["config"]
    except Exception:
        return None, None, None
