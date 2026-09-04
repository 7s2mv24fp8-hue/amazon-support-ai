"""
Amazon India Customer Support Executive
========================================

A rule-based + model-assisted support chatbot grounded in Amazon India's
official help guidelines.

Architecture:
  1. Intent classifier  - keyword/pattern matching to identify topic
  2. Response engine    - retrieves the precise policy answer for that topic
  3. Model fallback     - uses your GPT checkpoint for open-ended replies
  4. Conversation loop  - multi-turn, friendly, branded chat interface

Usage:
    python support_chat.py                        # rule-based only (no model needed)
    python support_chat.py --model checkpoints/   # use your GPT checkpoint as fallback
    python support_chat.py --debug                # show matched intent
"""

import re
import sys
import argparse
import textwrap
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (derived from Amazon India official help pages)
# ──────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "cannot_place_order": {
        "patterns": [
            r"(can'?t|cannot|unable|not able|problem|issue|trouble).*(place|make|do|complete|submit).*(order|purchase|buy)",
            r"(place|placing|make|making).*(order|purchase).*(problem|issue|fail|error)",
            r"why (am i|can'?t i|can i not).*(order|buy|purchase)",
            r"order.*(not going through|failing|failed|error)",
        ],
        "response": """I understand you're having trouble placing an order. Here are the most common reasons this happens:

  1. 💳 **Payment Decline** — Your bank may have declined the transaction. Try a different card or payment method.
  2. 📦 **Item Out of Stock** — The item may no longer be available.
  3. 🌐 **Technical/Connectivity Issue** — Check your internet connection and try refreshing.
  4. 📍 **Undeliverable Pincode** — Amazon may not deliver to your selected location.
  5. 🔢 **Quantity Limit** — Some items have a purchase limit per customer.
  6. ⏳ **FBA + Seller Items Combined** — Mixed carts can take extra time to process.

**Quick fix:** Try clearing your browser cache/app cache, check your pincode on the product page, and verify your payment method is valid.

Would you like help with any specific issue from the list above?"""
    },

    "modify_order": {
        "patterns": [
            r"(change|modify|update|edit|alter).*(order|item|quantity|number)",
            r"(order).*(change|modify|update|edit)",
            r"(add|remove).*(item|product).*(order)",
        ],
        "response": """Once an order is placed, **you cannot change or modify the items or quantity**.

Here's what you CAN do:
  ✅ **Cancel the order** and place a fresh one with the correct items.
  ✅ **Change shipping preferences** (address, delivery instructions) in Your Account — as long as the order hasn't entered shipping yet.

👉 To cancel: Go to **Your Orders → Select item → Cancel Items**

Would you like me to walk you through the cancellation process?"""
    },

    "cancel_order": {
        "patterns": [
            r"cancel.*(order|item|purchase)",
            r"(order|item).*(cancel|cancellation)",
            r"how (do i|to|can i).*(cancel)",
            r"want to cancel",
        ],
        "response": """Here's how to cancel your order:

**Before Shipment (easiest):**
  1. Go to → [Your Orders](https://www.amazon.in/gp/css/your-orders-access)
  2. Select the item you want to cancel
  3. Click **"Cancel Items"**
  4. Provide a reason (optional) and confirm

**After Shipment:**
  1. Go to → [Your Orders](https://www.amazon.in/gp/css/your-orders-access)
  2. Select **"Request Cancellation"**
  3. The item will be returned and you'll receive a full refund

💡 **Refunds** are processed back to your original payment method once the item is returned to Amazon."""
    },

    "track_order": {
        "patterns": [
            r"track.*(order|package|shipment|delivery)",
            r"(order|package|shipment).*(track|status|where|location)",
            r"where is my (order|package|parcel|shipment)",
            r"(status|update).*(order|delivery|shipment)",
        ],
        "response": """To track your order:

  1. Go to **Your Orders** on Amazon India
  2. Find the order you want to track
  3. Click **"Track Package"** next to the order
  4. Click **"See all Updates"** for detailed delivery updates

You'll also see the **courier partner name** (like Delhivery, Ekart, BlueDart, etc.) — you can track directly on their website using the tracking ID shown in Your Orders.

Is there anything specific about your delivery you'd like to know?"""
    },

    "late_delivery": {
        "patterns": [
            r"(late|delayed|delay|slow|overdue).*(delivery|order|shipment|package)",
            r"(delivery|order|shipment).*(late|delayed|delay|not arrived|hasn'?t arrived|overdue)",
            r"(expected|estimated).*(delivery|date).*(passed|gone|over)",
            r"not (received|delivered|arrived).*(yet|still)",
        ],
        "response": """I'm sorry to hear your delivery is running late! Here are common reasons for delays:

  📮 Incorrect or incomplete address
  🌧️ Severe weather conditions
  📦 Product damaged in transit
  🗺️ Local/regional delivery constraints

**What you should do:**
  1. ✅ **Track your package** in Your Orders — check the updated estimated delivery date
  2. ✅ **Verify your address** is complete and correct
  3. ⏳ **Wait 48 hours** — Amazon notifies customers about significant delays
  4. 📞 If it's been more than 48 hours past the estimated date, contact Amazon support

**Amazon's A-to-Z Guarantee** protects you for third-party seller orders on both timely delivery and item condition.

Shall I help you with anything else?"""
    },

    "delivered_not_received": {
        "patterns": [
            r"(shows|marked|status).*(delivered|delivery).*(but|however|yet|still).*(not|haven'?t|didn'?t).*(received|got|arrived)",
            r"(not|haven'?t|didn'?t).*(received|got).*(shows|marked|status).*(delivered)",
            r"says delivered but (i haven'?t|not) (received|got)",
            r"(package|order|parcel).*(missing|lost).*(delivered|delivery)",
            r"tracking shows delivered",
        ],
        "response": """I understand how frustrating this is! Here's what to do when tracking shows delivered but you haven't received your package:

**Step 1 — Verify your address**
  → Check Your Orders to confirm the delivery address was correct.

**Step 2 — Look around**
  → Check with household members, neighbors, building security, or mailroom.
  → Check around your front door, porch, gate, or any safe drop spots.

**Step 3 — Check Message Center**
  → Go to Amazon Message Center to see if someone accepted the package on your behalf.

**Step 4 — Wait up to 24 hours**
  → Sometimes delivery agents accidentally scan packages as "delivered" while still in transit.

**Step 5 — Contact Courier Partner**
  → Find courier details in Your Orders → Track Package
  → Keep your **Tracking ID** handy (found in Your Orders → scroll down)

**Step 6 — Contact Amazon**
  → If still unresolved after 24 hours, reach out to Amazon customer support for an A-to-Z Guarantee claim.

Can I help you with anything else?"""
    },

    "undeliverable": {
        "patterns": [
            r"(order|package|shipment).*(undeliverable|undelivered|could not be delivered|failed delivery)",
            r"(undeliverable|undelivered|failed delivery).*(order|package|shipment)",
            r"why is my order (undeliverable|undelivered|not delivered)",
            r"delivery (attempt|failed|unsuccessful)",
        ],
        "response": """Your order may be marked as undeliverable for one of these reasons:

  📵 No one was available to accept delivery after multiple attempts
  🏠 Incorrect address or mismatched pincode/area
  🚫 Recipient refused delivery
  🐕 Driver safety concern (e.g., presence of an aggressive dog)
  📋 Address label became unreadable during transit
  💥 Package was damaged in transit
  🌧️ Severe weather or regional contingency
  📞 Customer couldn't be contacted and no safe drop location was available
  🔒 Security/mailroom didn't accept the package
  🚪 Access code needed for automated entry

**What to do:**
  → Update your address: **Your Account → Your Addresses**
  → Add specific delivery instructions (e.g., leave with neighbor, call before arriving, gate code)
  → Track refund status in **Your Orders**

Would you like help updating your delivery address?"""
    },

    "damaged_defective": {
        "patterns": [
            r"(damaged|defective|broken|not working|faulty|defect|scratched|dented|torn|missing parts|wrong item|fake|counterfeit|expired|leaking|dead on arrival)",
            r"(received|got|delivered).*(wrong|damaged|broken|defective|different|fake|expired)",
            r"(product|item|order).*(damage|defect|issue|problem|wrong|broken|fake)",
            r"item.*(not matching|doesn'?t match|different|missing)",
        ],
        "response": """I'm sorry you received a damaged or defective product! This qualifies for a return/replacement:

**What counts as damaged/defective:**
  ❌ Not working / has cuts, tears, broken parts, dents, or scratches
  ❌ Seal broken or leakage
  ❌ Missing parts or accessories
  ❌ Wrong size, color, or item
  ❌ Doesn't match product description
  ❌ Product missing but box untampered
  ❌ Expired product
  ❌ Dead on arrival / screen damaged
  ❌ Fake or counterfeit product
  ❌ Correct box, incorrect item inside

**How to get a replacement or refund:**
  1. Go to **Your Orders**
  2. Select the affected order
  3. Click **"Return or Replace Items"**
  4. Choose your reason and follow the steps

You're also protected under **Amazon's A-to-Z Guarantee** for third-party seller items.

Would you like more help with your return?"""
    },

    "emi": {
        "patterns": [
            r"(emi|equated monthly installment|installment|monthly payment)",
            r"(pay|purchase|buy).*(installment|emi|monthly)",
            r"(no cost emi|zero cost emi|interest free)",
            r"(credit card|debit card|amazon pay later).*(emi|installment)",
        ],
        "response": """Here's everything you need to know about EMI on Amazon India:

**Who is eligible?**
  ✅ All credit card holders
  ✅ Debit card holders (check eligibility on Debit EMI page)
  ✅ Amazon Pay Later users

**Payment methods that support EMI:**
  💳 Credit Cards | 💳 Debit Cards | 📱 Amazon Pay Later | 🏦 Bajaj Finserv Cards

**No Cost EMI:**
  → The interest is pre-adjusted in the product price, so you pay **zero extra**. Total payable = Product price.

**How to pay via EMI:**
  1. Select the product → View all EMI Plans
  2. Choose your preferred payment method
  3. Select No Cost EMI (if available)
  4. Choose installment amount and tenure
  5. Review order summary and place order

⚠️ Note: The full amount may be temporarily deducted by your bank but will be converted to installments within **2–4 days**.

Do you need help with a specific EMI query?"""
    },

    "payment_failed": {
        "patterns": [
            r"(payment|transaction).*(fail|failed|failing|decline|declined|not (going through|processed|working))",
            r"(fail|failed|error|decline|declined).*(payment|transaction|order)",
            r"(couldn'?t|could not|unable to).*(pay|make payment|complete payment|process payment)",
            r"payment (issue|problem|error)",
        ],
        "response": """I'm sorry to hear your payment failed! Here are the most common reasons:

**Why payments fail:**
  🔢 Incorrect card details (number, CVV, expiry, name, 3D-Secure PIN)
  🏦 Bank technical issue or outage
  🔄 Page was closed/refreshed while payment was processing
  🚫 Card blocked for online transactions
  📊 Purchase outside your normal spending pattern (bank security block)
  💳 Card not accepted on Amazon India (check Accepted Payment Methods)
  🇺🇸 American Express: incorrect billing address or PIN code

**How to fix it:**
  1. Go to **Your Orders**
  2. Find the pending/failed order
  3. Click **"Revise Payment"** and follow instructions

💡 **Pro tip:** Try a different payment method — UPI, Net Banking, or a different card.

Need help with a specific payment method?"""
    },

    "upi_failed": {
        "patterns": [
            r"upi.*(fail|failed|not working|error|decline|declined|issue)",
            r"(fail|failed|error|issue).*(upi)",
            r"upi (payment|transaction|transfer)",
            r"(gpay|phonepe|paytm|bhim).*(fail|issue|error)",
        ],
        "response": """Here's why your UPI transaction may have failed and how to fix it:

**Common UPI failure reasons:**
  🆔 Incorrect UPI ID (format should be xyz@abc)
  ⏱️ Transaction timeout or server issue — retry after 20 minutes
  💰 Insufficient balance in bank account
  🔢 Daily limit exceeded: max ₹1,00,000 per day
  📊 Transaction count limits:
     → Max 10 Scan+Send transactions in 24 hours
     → Max ₹5,000 in 24 hours for new UPI registrations
     → Max 20 small transactions (₹10 or less) in 30 days
     → Max 100 Scan+Send transactions per month
  🏦 Bank server outage or technical glitch

**Important:** UPI is only available on the **Amazon India app** — not on mobile browser or desktop browser.

**What to do:**
  → Wait 20 minutes and retry via Your Orders page or the link in your email
  → Try a different payment method (Net Banking, Card, Amazon Pay)
  → Contact your bank for your specific UPI transaction limits

Would you like help choosing an alternative payment method?"""
    },

    "unknown_charge": {
        "patterns": [
            r"(unknown|unauthorized|unexpected|strange|weird|unrecognized).*(charge|deduction|transaction|payment)",
            r"(charge|deduction|transaction|payment).*(unknown|unauthorized|unexpected|strange|unrecognized)",
            r"(money|amount|charge).*(deducted|taken|removed).*(without|unknowingly|unexpectedly)",
            r"why (was i|am i).*(charged|debited)",
        ],
        "response": """I understand seeing an unexpected charge is concerning. Before reporting it as unauthorized, please check:

**Common reasons for unfamiliar charges:**
  👨‍👩‍👧 Family member placed an order using your card
  🔗 Another card linked to your account was used
  📦 Pre-order item was charged when it became available
  🔄 Cancelled/changed order — some banks show authorizations as charges
  🛒 Amazon Pay purchase on an external merchant website
  ⭐ **Prime subscription auto-renewal** — check if auto-pay is enabled

**How to verify:**
  → Check complete order history in **Your Account**
  → Visit **Manage Your Prime Membership** for subscription charges
  → Check **Amazon Pay** transaction history for external purchases

**To report an unauthorized transaction:**
  📞 Call Amazon India: **1800-1200-1637**

**For failed/pending transaction issues:**
  → Visit the Payment Issues help page or use Revise Payment in Your Orders

Shall I help you investigate the charge further?"""
    },

    "returns_refund": {
        "patterns": [
            r"(return|refund|replace|replacement|exchange).*(order|product|item|package)",
            r"(order|product|item|package).*(return|refund|replace|replacement|exchange)",
            r"how (do i|to|can i).*(return|refund|replace|exchange)",
            r"(want to|need to).*(return|refund|get refund)",
            r"refund (status|when|timeline|time)",
        ],
        "response": """Here's how to return an item and get a refund on Amazon India:

**How to Return:**
  1. Go to **Your Orders**
  2. Select the order with the item to return
  3. Click **"Return or Replace Items"**
  4. Select your reason and choose pickup or drop-off

**Refund Timelines (after return is received by Amazon):**
  | Payment Method      | Refund Time          |
  |---------------------|----------------------|
  | Credit/Debit Card   | 3–5 business days    |
  | Net Banking         | 3–5 business days    |
  | UPI                 | 2–4 business days    |
  | Amazon Pay Balance  | Instant              |
  | Amazon Gift Card    | 3–5 business days    |

💡 Track your refund status in **Your Orders** → Select order → View refund details.

Do you have a specific return or refund question?"""
    },

    "prime": {
        "patterns": [
            r"(prime|prime membership|prime subscription)",
            r"(cancel|manage|renew|renewal).*(prime|membership|subscription)",
            r"prime.*(benefits|cancel|renew|cost|price|charged)",
        ],
        "response": """Here's how to manage your Amazon Prime membership:

**To manage Prime:**
  → Visit **Manage Your Prime Membership** on Amazon India
  → From there you can: view benefits, check renewal date, update payment method, or cancel

**Prime Auto-Renewal:**
  → Prime renews automatically. If you see an unexpected charge, it may be your annual or monthly renewal.
  → To turn off auto-renewal, go to Manage Your Prime Membership → End Membership

**Prime Benefits include:**
  🚀 Fast delivery | 🎬 Prime Video | 🎵 Prime Music | 📚 Prime Reading | 🛒 Exclusive deals

Do you need help with a specific Prime issue?"""
    },

    "greeting": {
        "patterns": [
            r"^(hi|hello|hey|good morning|good afternoon|good evening|namaste|hii|helo|helloo|sup|yo)[\s!.,]*$",
            r"^(hi|hello|hey).*(there|amazon|support|help)[\s!.,]*$",
        ],
        "response": None  # handled dynamically
    },

    "goodbye": {
        "patterns": [
            r"^(bye|goodbye|see you|thanks bye|thank you bye|that'?s all|nothing else|no that'?s all|all good now|resolved|sorted)[\s!.,]*$",
            r"(thank you|thanks).*(that'?s all|nothing else|goodbye|bye|that will be all)",
        ],
        "response": None  # handled dynamically
    },

    "thanks": {
        "patterns": [
            r"^(thank you|thanks|thank u|thx|ty|tysm|thank you so much|many thanks)[\s!.,]*$",
            r"(thank you|thanks).*(help|assistance|support|info|information)",
        ],
        "response": None  # handled dynamically
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def classify_intent(user_input: str) -> str:
    """Match user input to an intent using regex patterns."""
    text = user_input.lower().strip()
    for intent, data in KNOWLEDGE_BASE.items():
        for pattern in data["patterns"]:
            if re.search(pattern, text):
                return intent
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# RESPONSE GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class SupportExecutive:
    """Amazon India Customer Support Executive."""

    AGENT_NAME = "Priya"
    BRAND = "Amazon India"

    def __init__(self, model=None, tokenizer=None, config=None, debug=False):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.debug = debug
        self.conversation_history = []
        self.customer_name = None
        self.session_start = datetime.now()
        self.turn_count = 0

    def _greeting(self) -> str:
        name_part = f", {self.customer_name}" if self.customer_name else ""
        return (
            f"Hello{name_part}! 👋 I'm **{self.AGENT_NAME}** from {self.BRAND} Customer Support.\n"
            f"I'm here to help you with orders, payments, deliveries, returns, and more.\n\n"
            f"How can I assist you today?"
        )

    def _farewell(self) -> str:
        name_part = f", {self.customer_name}" if self.customer_name else ""
        return (
            f"Thank you for contacting {self.BRAND} Support{name_part}! 😊\n"
            f"I hope your issue has been fully resolved. Have a wonderful day and happy shopping! 🛒\n\n"
            f"*This session has ended. Type anything to start a new conversation.*"
        )

    def _thanks(self) -> str:
        return (
            f"You're very welcome! 😊 I'm always happy to help.\n"
            f"Is there anything else I can assist you with regarding your Amazon orders or account?"
        )

    def _unknown(self, user_input: str) -> str:
        model_response = self._try_model_generate(user_input)
        if model_response:
            return model_response
        return (
            f"I'm sorry, I didn't quite understand that. I can help you with:\n\n"
            f"  🛒 **Orders** — placing, tracking, cancelling, modifying\n"
            f"  📦 **Deliveries** — late, undeliverable, missing packages\n"
            f"  💳 **Payments** — failures, EMI, UPI, unknown charges\n"
            f"  🔄 **Returns & Refunds** — process and timelines\n"
            f"  ⭐ **Prime Membership** — manage, cancel, benefits\n"
            f"  🛠️ **Damaged/Defective Products** — what qualifies, how to claim\n\n"
            f"Could you please describe your issue in a bit more detail? I'm here to help!"
        )

    def _try_model_generate(self, prompt: str) -> str | None:
        """Use the GPT checkpoint as a fallback for open-ended responses."""
        if self.model is None or self.tokenizer is None:
            return None
        try:
            import torch
            context = (
                f"Amazon India Customer Support conversation.\n"
                f"Customer: {prompt}\n"
                f"Support Agent {self.AGENT_NAME}:"
            )
            ids = self.tokenizer.encode(context)
            x = torch.tensor([ids], dtype=torch.long)
            device = next(self.model.parameters()).device
            x = x.to(device)
            self.model.eval()
            with torch.no_grad():
                block_size = self.config.get("block_size", 256)
                max_new = min(150, block_size - len(ids))
                for _ in range(max_new):
                    x_cond = x[:, -block_size:]
                    logits, _ = self.model(x_cond)
                    logits = logits[:, -1, :]
                    probs = torch.softmax(logits / 0.8, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                    x = torch.cat([x, next_id], dim=1)
                    decoded = self.tokenizer.decode([next_id.item()])
                    if decoded in ["\n\n", "Customer:", "Q:"]:
                        break
            generated = self.tokenizer.decode(x[0].tolist()[len(ids):])
            generated = generated.split("\n\n")[0].split("Customer:")[0].strip()
            if len(generated) > 30:
                return generated
        except Exception:
            pass
        return None

    def respond(self, user_input: str) -> str:
        """Generate a support response for the given user input."""
        self.turn_count += 1
        self.conversation_history.append({"role": "customer", "text": user_input})

        # Extract customer name if introduced
        name_match = re.search(
            r"(?:i(?:'m| am)|my name is|this is|call me)\s+([A-Z][a-z]+)",
            user_input,
            re.IGNORECASE,
        )
        if name_match and not self.customer_name:
            self.customer_name = name_match.group(1).capitalize()

        intent = classify_intent(user_input)

        if self.debug:
            print(f"  [DEBUG] Intent: {intent}")

        # Route to response
        if intent == "greeting":
            response = self._greeting()
        elif intent == "goodbye":
            response = self._farewell()
        elif intent == "thanks":
            response = self._thanks()
        elif intent == "unknown":
            response = self._unknown(user_input)
        else:
            response = KNOWLEDGE_BASE[intent]["response"]

        self.conversation_history.append({"role": "agent", "text": response})
        return response


# ──────────────────────────────────────────────────────────────────────────────
# TERMINAL UI
# ──────────────────────────────────────────────────────────────────────────────

def print_banner():
    """Print the Amazon support chat banner."""
    width = 70
    print("\n" + "═" * width)
    print("  🛒  AMAZON INDIA  |  Customer Support Executive")
    print("  Powered by AI — Grounded in Amazon Official Guidelines")
    print("═" * width)
    print("  Type your question below. Type 'quit' or 'exit' to end the session.")
    print("═" * width + "\n")


def format_agent_response(agent_name: str, text: str) -> str:
    """Format the agent response for terminal display."""
    lines = text.split("\n")
    formatted = []
    for line in lines:
        if line.startswith("  ") or line.startswith("  "):
            formatted.append(line)
        else:
            formatted.append(line)
    output = "\n".join(formatted)
    return f"\n\033[96m🤖 {agent_name}:\033[0m\n{output}\n"


def format_customer_input(name: str | None) -> str:
    label = name if name else "You"
    return f"\033[93m👤 {label}:\033[0m "


def load_model_from_checkpoint(ckpt_dir: str):
    """Load GPT model and tokenizer from a checkpoint directory."""
    try:
        import torch
        from model import GPT
        from tokenizer import CharTokenizer

        ckpt = torch.load(f"{ckpt_dir}/model.pt", map_location="cpu")
        tok = CharTokenizer.load(f"{ckpt_dir}/vocab.json")
        model = GPT(**ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"✅ Loaded GPT model from '{ckpt_dir}' ({ckpt['config']['n_layer']} layers, "
              f"{ckpt['config']['n_embd']} embd dim)")
        return model, tok, ckpt["config"]
    except FileNotFoundError:
        print(f"⚠️  No checkpoint found at '{ckpt_dir}'. Running in rule-based mode only.")
        return None, None, None
    except ImportError as e:
        print(f"⚠️  Could not import model modules: {e}. Running in rule-based mode only.")
        return None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Amazon India Support Executive Chatbot")
    parser.add_argument("--model", default=None, metavar="CKPT_DIR",
                        help="Path to GPT checkpoint directory for model-assisted fallback")
    parser.add_argument("--debug", action="store_true",
                        help="Show matched intent for each message")
    args = parser.parse_args()

    # Load model if provided
    model, tokenizer, config = None, None, None
    if args.model:
        model, tokenizer, config = load_model_from_checkpoint(args.model)

    # Initialise support executive
    agent = SupportExecutive(model=model, tokenizer=tokenizer, config=config, debug=args.debug)

    print_banner()

    # Opening greeting
    opening = agent.respond("hello")
    print(format_agent_response(agent.AGENT_NAME, opening))

    # Conversation loop
    session_active = True
    while session_active:
        try:
            user_input = input(format_customer_input(agent.customer_name)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            print(format_agent_response(agent.AGENT_NAME, agent._farewell()))
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print(format_agent_response(agent.AGENT_NAME, agent._farewell()))
            session_active = False
            continue

        response = agent.respond(user_input)
        print(format_agent_response(agent.AGENT_NAME, response))

        # End session after farewell
        if classify_intent(user_input) == "goodbye":
            session_active = False

    # Session summary
    print("\n" + "─" * 70)
    print(f"  Session ended | Turns: {agent.turn_count} | "
          f"Duration: {(datetime.now() - agent.session_start).seconds}s")
    print("─" * 70 + "\n")


if __name__ == "__main__":
    main()
