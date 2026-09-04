"""
self_play_trainer.py — Auto-generate Amazon India Support Conversations
=========================================================================

Simulates realistic conversations between a "Customer" and an "Amazon Support
Expert", then saves them as training files for continual_learn.py to absorb.

Pipeline:
    Customer Simulator  ──► Expert Agent  ──► incoming/*.txt  ──► continual_learn.py  ──► GPT model

Three modes (auto-detected, best → fallback):
    1. Ollama  (100% free, local LLM — best quality)
       → Install: https://ollama.com  then: ollama pull llama3.2
    2. Groq    (free API tier — llama3-70b quality)
       → Set env:  export GROQ_API_KEY=your_key
    3. Template (zero dependency — built-in, always works)

Usage:
    python3 self_play_trainer.py                        # 1 batch of 20 conversations
    python3 self_play_trainer.py --num 50               # generate 50 conversations
    python3 self_play_trainer.py --watch                # keep generating + training in a loop
    python3 self_play_trainer.py --mode template        # force template mode
    python3 self_play_trainer.py --mode ollama          # force ollama mode
    python3 self_play_trainer.py --mode groq            # force groq mode
    python3 self_play_trainer.py --watch --train        # generate AND auto-train each batch
    python3 self_play_trainer.py --ollama_model llama3.2:1b  # use a smaller model
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from textwrap import dedent


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOMER TEMPLATES
# ──────────────────────────────────────────────────────────────────────────────

CUSTOMER_SCENARIOS = [
    {
        "topic": "cannot_place_order",
        "variations": [
            "i am unable to place an order on amazon. what should i do?",
            "why cant i place my order? it keeps failing",
            "order not going through, tried multiple times",
            "getting error while placing order",
            "amazon not letting me buy anything",
            "my order keeps getting rejected",
            "i am trying to buy a product but cannot complete the purchase",
            "the buy now button is not working for me",
            "i tried to place order 3 times but it keeps showing error",
            "cant complete checkout on amazon india",
            "order submission failing repeatedly what do i do",
            "why am i not able to place orders suddenly",
        ]
    },
    {
        "topic": "payment_failed",
        "variations": [
            "my payment failed but money got deducted from my account",
            "payment declined on amazon what to do",
            "i paid but order not confirmed yet",
            "my credit card payment is not going through",
            "payment keeps failing every time i try",
            "transaction failed but amount deducted from bank",
            "debit card payment failing on amazon",
            "payment error code showing after i enter card details",
            "my order shows payment pending",
            "amazon not accepting my card",
            "payment gateway error on checkout",
            "my net banking payment failed but money got cut",
        ]
    },
    {
        "topic": "upi_failed",
        "variations": [
            "my upi payment failed on amazon",
            "gpay transaction failed but money deducted",
            "phonepe payment not working on amazon india",
            "upi payment timed out what happens now",
            "bhim upi transaction declined",
            "my upi id is correct but payment still failing",
            "upi limit exceeded message showing",
            "amazon upi option not available on browser",
            "upi transaction shows failed in gpay but pending on amazon",
            "paytm upi payment declined by bank",
        ]
    },
    {
        "topic": "track_order",
        "variations": [
            "how do i track my amazon order",
            "where is my package? delivery is taking too long",
            "i want to see my order status",
            "how to check delivery status of my order",
            "my order shows out for delivery but not arrived yet",
            "how do i know which courier has my package",
            "track package option not showing for my order",
            "i want to see live tracking for my delivery",
            "order status not updating since yesterday",
            "how to find tracking id of my amazon order",
        ]
    },
    {
        "topic": "late_delivery",
        "variations": [
            "my order was supposed to arrive yesterday but still not here",
            "delivery is delayed by 3 days, what should i do",
            "estimated delivery date has passed and no update",
            "my package is stuck at hub for 2 days",
            "amazon delivery is very late, i need it urgently",
            "shipment not moving for 4 days, what is happening",
            "my product hasnt arrived even after expected delivery date",
            "courier not attempting delivery even after 2 days",
            "order status shows in transit for 5 days now",
            "why is my delivery so delayed",
            "my package has been out for delivery since morning but not delivered yet",
        ]
    },
    {
        "topic": "delivered_not_received",
        "variations": [
            "amazon says my order is delivered but i never received it",
            "tracking shows delivered yesterday but no package at my door",
            "my order is marked delivered but i did not get it",
            "package shows delivered but not found anywhere",
            "i got an sms saying delivered but nothing received",
            "order status delivered but no delivery happened",
            "delivery boy marked delivered without actually delivering",
            "my order was delivered to wrong address",
            "someone else signed for my package and now its missing",
            "shows delivered 2 days ago but still no package",
        ]
    },
    {
        "topic": "cancel_order",
        "variations": [
            "i want to cancel my order how to do it",
            "can i cancel an order after placing it",
            "how to cancel amazon order before delivery",
            "i placed order by mistake, can i cancel",
            "cancel option not showing for my order",
            "my order has been shipped, can i still cancel",
            "i want to return the order before delivery",
            "how long does it take to cancel amazon order",
            "cancel button is greyed out for my order",
            "i accidentally ordered same item twice, cancel one",
        ]
    },
    {
        "topic": "returns_refund",
        "variations": [
            "how do i return a product on amazon india",
            "i want to return this item, what is the process",
            "when will i get my refund after returning",
            "return request not being accepted",
            "i returned the product but refund not received",
            "refund is taking too long what to do",
            "can i exchange this product instead of returning",
            "my return pickup is not scheduled yet",
            "how many days for refund after return",
            "amazon is not accepting return for this item",
            "return window expired but product is defective",
            "refund showing initiated but not credited to bank",
        ]
    },
    {
        "topic": "damaged_defective",
        "variations": [
            "i received a damaged product from amazon",
            "the item i received is broken",
            "wrong product was delivered to me",
            "i got a completely different item than what i ordered",
            "product seal was already broken when i received it",
            "i received an empty box, product is missing",
            "the item stopped working after 1 day, dead on arrival",
            "i got a duplicate or fake product",
            "screen of my phone was cracked when i opened the box",
            "i received an expired product",
            "parts are missing from my product",
            "the product i received has scratch marks on it",
        ]
    },
    {
        "topic": "modify_order",
        "variations": [
            "can i change quantity of my order",
            "i want to add another item to my existing order",
            "can i change the color i ordered",
            "i want to change the size i selected during checkout",
            "can i modify delivery address after ordering",
            "is it possible to change my order to a different variant",
            "i made a mistake in order, want to edit it",
        ]
    },
    {
        "topic": "undeliverable",
        "variations": [
            "my order was returned to seller marked as undeliverable",
            "delivery agent did not attempt delivery and marked undeliverable",
            "amazon says delivery failed due to no recipient available",
            "my order is marked as undeliverable, what happens now",
            "delivery was attempted but nobody was home, order cancelled",
            "i missed the delivery, is it gone forever",
            "courier returned the package without even trying to deliver",
        ]
    },
    {
        "topic": "emi",
        "variations": [
            "how do i pay in emi on amazon",
            "is emi available for my credit card",
            "what is no cost emi on amazon",
            "can i use debit card for emi",
            "emi option not showing at checkout",
            "how many months emi is available",
            "i want to convert my purchase to emi after ordering",
            "what is the interest rate for amazon emi",
            "can i use amazon pay later for emi",
        ]
    },
    {
        "topic": "unknown_charge",
        "variations": [
            "i see an unknown charge on my account from amazon",
            "money was deducted from my card without any order",
            "amazon charged me twice for same order",
            "i got a refund but full amount not returned",
            "why was i charged extra delivery fees",
            "my prime renewal charge is wrong",
            "i cancelled my prime but still got charged",
            "unauthorized transaction from amazon on my card",
        ]
    },
    {
        "topic": "prime",
        "variations": [
            "how do i cancel amazon prime",
            "how to stop prime auto renewal",
            "i dont want prime, how to unsubscribe",
            "my prime benefits are not showing",
            "prime delivery is not showing as free for me",
            "i subscribed to prime but video not working",
            "can i get refund on prime membership",
        ]
    },
    {
        "topic": "general",
        "variations": [
            "how do i contact amazon customer care",
            "what is amazon customer care number",
            "i need help with my account",
            "amazon india helpline number please",
            "how to raise a complaint on amazon",
            "i am very frustrated with amazon service",
            "please help me, i have been waiting for 7 days",
            "amazon is not responding to my emails",
        ]
    },
]

FOLLOWUP_QUESTIONS = [
    "how long will this take?",
    "can i get a faster resolution?",
    "i already tried that, it didnt work",
    "what if the issue doesnt resolve?",
    "will i get a full refund?",
    "who do i contact next?",
    "i am very disappointed with this experience",
    "can i speak to a senior executive?",
    "is there a complaint number?",
    "this has happened multiple times now",
    "my order number is 408-{order_id}-1234567",
    "i have already waited {days} days",
    "ok i will try that. thank you",
    "that worked! thank you so much",
    "is there anything else i can do?",
]

EXPERT_ANSWERS = {
    "cannot_place_order": """Thank you for reaching out to Amazon India Customer Support. I understand you're having trouble placing an order, and I'm here to help.

There are several common reasons why orders may not go through:

1. Payment Decline - Your bank may have declined the transaction. I'd recommend trying a different card or payment method such as UPI, Net Banking, or Amazon Pay Balance.

2. Item Out of Stock - The item may have gone out of stock after you added it to your cart. Please refresh the product page to confirm availability.

3. Technical or Connectivity Issue - Please clear your browser cache or app cache, and try again. If using the app, try force-closing and reopening it.

4. Undeliverable Pincode - Amazon may not deliver certain items to your pincode. You can check serviceability by entering your pincode on the product page.

5. Quantity Limit - Some items have a per-customer purchase limit.

6. Mixed Cart Processing - When Fulfilled by Amazon and seller-fulfilled items are in the same cart, processing may take longer.

Could you please tell me which specific error message you are seeing? That will help me give you a more precise solution.""",

    "payment_failed": """I completely understand how frustrating it is when a payment fails. Let me help you resolve this right away.

Here are the most common reasons for payment failures on Amazon India:

- Incorrect card details: Please double-check your card number, expiry date, CVV, and the name on the card.
- 3D Secure PIN error: Make sure you are entering the OTP or 3D Secure PIN correctly.
- Card blocked for online transactions: Some banks block cards for online transactions by default. Please contact your bank to enable online transactions.
- Bank outage: Your bank's payment servers may be temporarily down. Try again after 15-30 minutes.
- Browser issue: If you closed or refreshed the page while payment was being processed, this can cause a failure.

To revise payment for a pending order:
1. Go to Your Orders on Amazon India
2. Find the order with payment pending
3. Click Revise Payment and complete the payment using a different method

If money was deducted but order is not confirmed, please do not worry. The amount will be automatically refunded to your account within 5-7 business days. Would you like me to help you with anything else?""",

    "upi_failed": """I understand your UPI payment has failed, and I'm sorry for the inconvenience. Let me explain what may have happened and how to resolve it.

Common reasons for UPI payment failures:

1. Incorrect UPI ID: Please verify your UPI ID is correct. The format is usually yourname@bankname or number@upi.
2. Transaction timeout: UPI payments have a short window. If it timed out, please wait 20 minutes before retrying.
3. Daily transaction limits: UPI has a daily limit of Rs 1,00,000 per day. You may also have a bank-specific lower limit.
4. Insufficient balance: Please ensure you have enough balance in your linked bank account.
5. New UPI registration: If you recently registered, you can only send up to Rs 5,000 in the first 24 hours.

Important note: UPI payments on Amazon are only supported in the Amazon India mobile app. UPI is NOT available on the mobile browser or desktop website.

If the payment failed but money was deducted, it will be automatically reversed within 24-48 hours. You can try completing the order using Net Banking or a debit or credit card in the meantime. Would you like to try an alternative payment method?""",

    "track_order": """I will be happy to help you track your order. Here are the steps:

1. Open the Amazon India app or website
2. Go to Your Orders in the menu or top navigation
3. Find the order you want to track
4. Click Track Package next to your order
5. Click See all Updates for detailed, step-by-step delivery progress

You will also see the name of your courier partner such as Delhivery, Ekart, BlueDart, or DTDC along with their tracking ID. You can use this tracking ID to track your package directly on the courier's website for more real-time updates.

If the Track Package option is not visible, it usually means the seller has not yet dispatched your order. Orders are typically dispatched within 1-2 business days of placement.

Is there a specific order you are trying to track? If you share the order number, I can guide you more specifically.""",

    "late_delivery": """I sincerely apologize for the delay in your delivery. I understand how important it is to receive your order on time, and I want to help resolve this for you.

Deliveries can get delayed due to several reasons including high delivery volume in your area, weather conditions, regional disturbances, incorrect or incomplete delivery address, or delivery attempts were made but nobody was available at home.

Here is what I recommend:

Step 1: Track your package in Your Orders to see the latest status and estimated delivery date. Amazon updates this information whenever there is a change.

Step 2: Verify your delivery address is complete, including the flat number, building name, landmark, and pincode.

Step 3: Wait up to 48 hours beyond the estimated delivery date, as Amazon typically notifies customers about significant delays via SMS and email.

Step 4: If it has been more than 48 hours past the estimated date with no update, please contact Amazon customer support and we will investigate immediately.

I also want to assure you that you are protected under Amazon's A-to-Z Guarantee for third-party seller orders. If your order does not arrive, you will receive a full refund. Is there anything specific I can check for your order?""",

    "delivered_not_received": """I completely understand how concerning this is, and I sincerely apologize for the inconvenience. Let me help you investigate this right away.

Please follow these steps:

Step 1 - Verify your delivery address. Go to Your Orders and confirm that the delivery address on your order is correct and matches your current address.

Step 2 - Look around your delivery location. Sometimes packages are left at the doorstep, with building security, at a neighbor's place, or in a mailroom. Please check these locations.

Step 3 - Check with household members. Ask if someone else at your home or office accepted the package on your behalf.

Step 4 - Check your Amazon Message Center. Go to the Message Center to see if the delivery partner left a message about where they left the package.

Step 5 - Wait up to 24 hours. In some cases, delivery agents accidentally scan packages as delivered while they are still in transit. The package often arrives the next day.

Step 6 - Contact the courier partner. In Your Orders, click Track Package to find the courier partner name and their tracking ID. Contact the courier directly with your tracking number.

If after all these steps you still have not received your package, please contact Amazon and we will raise an investigation and ensure you receive either a replacement or a full refund. Your satisfaction is our priority.""",

    "cancel_order": """Of course, I can help you cancel your order. Here is what you need to do:

If your order has NOT yet been shipped:
1. Go to Your Orders on Amazon India
2. Select the item you want to cancel
3. Click Cancel Items
4. Select a reason for cancellation which is optional
5. Confirm the cancellation

The cancellation will be immediate and if any payment was made, a full refund will be processed to your original payment method.

If your order has ALREADY been shipped:
1. Go to Your Orders
2. Select the relevant order
3. Click Request Cancellation
4. The delivery agent will attempt to collect the package, or you can refuse delivery when it arrives
5. A full refund will be processed once the item is returned to Amazon

Refund timelines after cancellation:
- Credit or Debit Card: 3-5 business days
- UPI: 2-4 business days
- Amazon Pay Balance: Instant
- Net Banking: 3-5 business days

Please note that once an order is shipped and delivered, you cannot cancel it but you can initiate a return. Would you like help with anything else?""",

    "returns_refund": """I will be happy to help you with the return process. Here are the complete steps:

How to Return a Product:
1. Go to Your Orders on Amazon India
2. Find the order containing the item you want to return
3. Click Return or Replace Items
4. Select the item and choose your reason for return
5. Choose your preferred return method: Pickup where courier will come to your address, or Drop-off where you drop it at a nearby hub
6. Schedule the pickup or get the drop-off location details

Refund Timelines after Amazon receives your returned item:
- Credit Card or Debit Card: 3 to 5 business days
- Net Banking: 3 to 5 business days
- UPI: 2 to 4 business days
- Amazon Pay Balance: Instant
- Gift Card: 3 to 5 business days

Important points:
- Most items have a 10-day return window from the delivery date
- Some items like perishables, digital content, and certain electronics have different policies
- You can track your refund status in Your Orders

If your return window has expired but the product is defective or damaged, please contact Amazon customer support immediately. We have exceptions for defective products. Is there anything else I can help you with?""",

    "damaged_defective": """I am so sorry to hear that you received a damaged or defective product. This is certainly not the experience we want for our customers, and I assure you this will be resolved.

What qualifies for a damage or defect claim:
- Product is not in working condition or has visible damage
- Product seal was broken or there is leakage
- Parts or accessories are missing
- Wrong product or wrong variant was delivered
- Expired product received
- Fake or counterfeit product
- Empty box received

How to get a replacement or refund:
1. Go to Your Orders on Amazon India
2. Select the affected order
3. Click Return or Replace Items
4. Select the item and choose Defective or Damaged as the reason
5. Choose between a replacement or a refund
6. Schedule a pickup for the damaged item

You are also protected under Amazon's A-to-Z Guarantee for items sold by third-party sellers.

I want to assure you that if a product arrives damaged or defective, Amazon takes full responsibility regardless of who sold it. We will make sure you either receive a perfect replacement or a full refund. Would you like me to walk you through anything specific?""",

    "modify_order": """Thank you for your question. I understand you would like to make changes to your order.

Unfortunately, once an order is placed on Amazon India, it is not possible to modify the items, quantity, size, color, or any product details. This is because order processing begins immediately after placement.

However, here is what you can do:

Option 1 - Cancel and Reorder if not yet shipped:
1. Go to Your Orders
2. Cancel the existing order by selecting Cancel Items
3. Place a fresh new order with the correct items or specifications

Option 2 - Change Delivery Address if order not yet shipped:
1. Go to Your Orders
2. Select the order
3. Look for the option to change the shipping address

Option 3 - Return after Delivery if already shipped:
Once you receive the item, you can return it through Your Orders and place a new order for the correct item.

I would recommend checking Your Orders first to see if the cancellation option is still available. Is there anything specific you would like help with?""",

    "undeliverable": """I understand how frustrating it is when a delivery attempt fails. Let me explain what happened and what we can do.

Your order may have been marked undeliverable for one of these reasons:
- No one was available to accept delivery after multiple attempts
- The delivery address was incorrect or incomplete
- The pincode and area mentioned did not match
- Security or mailroom personnel declined to accept the package
- Severe weather conditions prevented delivery

What happens next:
When an order is marked undeliverable, the package is typically returned to the seller or Amazon warehouse. In this case, you will automatically receive a full refund to your original payment method within 5-7 business days.

To avoid this in the future:
1. Go to Your Account and update your delivery address with complete details including flat number, building name, and landmarks
2. Add a delivery note such as leave with security if unavailable or call before delivery
3. Make sure your phone number on the order is correct and reachable

If you would like to receive the item, you can simply place a new order with an updated and complete address. Would you like help with anything else?""",

    "emi": """Great question. Let me explain the EMI options available on Amazon India.

Who can use EMI:
- All Credit Card holders including Visa, Mastercard, Amex, and RuPay
- Debit Card holders, check eligibility on Amazon
- Amazon Pay Later users
- Bajaj Finserv cardholders

How to pay using EMI:
1. Add the product to your cart and go to checkout
2. Select your payment method such as Credit Card, Debit Card, or Amazon Pay Later
3. Look for the Pay with EMI option and select it
4. View available EMI plans for 3 months, 6 months, 9 months, 12 months, and more
5. Select the No Cost EMI tab if available to see interest-free options
6. Choose your preferred tenure and confirm

About No Cost EMI:
Under No Cost EMI, the interest amount is pre-adjusted in the product price. You pay zero extra cost. The total amount payable including all charges equals the product price.

For regular EMI, your bank will charge interest. The interest amount will be shown clearly before you confirm.

Note: After placing the order, the full amount may be temporarily deducted by your bank but it will be converted to monthly installments within 2-4 business days.

Would you like help finding if a specific product has No Cost EMI available?""",

    "unknown_charge": """I understand seeing an unexpected charge can be alarming. Let me help you identify what this charge could be.

Before reporting it as unauthorized, please check the following:

1. Family member order: Check if a family member who has access to your account or card may have placed an order.

2. Pre-order fulfillment: If you placed a pre-order, the amount gets charged when the product becomes available.

3. Amazon Prime renewal: Check if your Prime membership auto-renewed. Visit Manage Your Prime Membership to see your renewal date and charges.

4. Amazon Pay transactions: Check if any payment was made on external websites using Amazon Pay.

5. Pending authorizations: Sometimes cancelled orders show as temporary charges. These reverse automatically within 5-7 days.

6. Order outside Amazon.in: Check if any purchase was made on Amazon.com or Amazon Global.

To view your complete order history and charges, go to Your Account and then Your Orders which includes all purchases. Also go to Your Account and then Amazon Pay and then Transaction History for all Amazon Pay transactions.

If after checking all of this you still see an unauthorized charge, please call Amazon India customer support immediately at 1800-1200-1637 which is toll free. We will investigate and ensure your money is protected. Is there anything else I can help clarify?""",

    "prime": """I will be happy to help you with your Amazon Prime membership. Here is what you need to know:

To cancel Amazon Prime:
1. Go to Manage Your Prime Membership on Amazon India
2. Click End Membership and Benefits
3. Follow the on-screen instructions to confirm cancellation

To stop auto-renewal:
1. Go to Manage Your Prime Membership
2. Select Do not continue or disable auto-renewal
3. Your Prime benefits will remain active until the end of your current billing cycle

About refunds for Prime:
- If you cancel within 3 days of renewal and have not used any Prime benefits, you may be eligible for a full refund
- If you have used Prime benefits, a prorated refund may be available

Prime benefits include free and fast delivery on eligible items, Prime Video for movies and web series, Prime Music, Prime Reading, and exclusive deals and early access to lightning deals.

If your Prime benefits are not working correctly despite being a member, please try signing out and signing back in to your Amazon account. If the issue persists, contact Amazon customer support. Is there anything specific about your Prime membership I can help with?""",

    "general": """Thank you for contacting Amazon India Customer Support. I am Priya and I am here to assist you.

You can reach Amazon India customer support through the following channels:

1. Online Chat: Go to amazon.in then Help then Contact Us then Chat with us
2. Phone: Call 1800-1200-1637 which is toll free and available 24 hours a day 7 days a week
3. Email or Callback: Go to Help then Contact Us then Request a callback or email response
4. Amazon App: Open the app then Menu then Customer Service

I can also help you directly right now. I am trained to assist with:
- Order issues including placing, tracking, cancelling, and modifying
- Payment problems including failures, UPI issues, unknown charges, and EMI
- Delivery issues including delays, missing packages, and undeliverable orders
- Returns and refunds
- Damaged or defective products
- Amazon Prime membership
- Account issues

Please tell me what issue you are facing and I will do my best to help you resolve it as quickly as possible. Your satisfaction is our top priority."""
}

CLOSING_RESPONSES = [
    "I am glad I could assist you today. Your satisfaction is our top priority at Amazon India. If you face any other issues with your orders, payments, or deliveries, please do not hesitate to reach out to us. We are available 24 hours a day at 1800-1200-1637. Have a wonderful day and happy shopping.",
    "Thank you for your patience and for choosing Amazon India. I hope this has resolved your concern. Please feel free to contact us anytime if you need further assistance. Is there anything else I can help you with today?",
    "I completely understand, and I apologize for the inconvenience. Please rest assured that Amazon will take care of this for you. You can track the status in Your Orders at any time. Thank you for shopping with Amazon India.",
    "I am happy to confirm that your issue has been noted and will be resolved at the earliest. You will receive an SMS and email update. Thank you for your patience, and please do not hesitate to contact us again if needed. Have a great day.",
]

INTERMEDIATE_RESPONSES = [
    "I completely understand your frustration, and I sincerely apologize for this inconvenience. Let me look into this further for you.",
    "Thank you for providing those details. Based on what you have shared, I would recommend the following next steps.",
    "I hear you, and I assure you this will be resolved. Please give us a little more time to investigate this.",
    "I apologize for the delay. Your concern has been escalated and you will receive an update within 24-48 hours.",
    "Thank you for your patience. I have noted your order details and our team will ensure this is resolved at the earliest.",
]


# ──────────────────────────────────────────────────────────────────────────────
# LLM BACKENDS
# ──────────────────────────────────────────────────────────────────────────────

AMAZON_SYSTEM_PROMPT = """You are Priya, a senior customer support executive at Amazon India.
You are warm, empathetic, professional, and extremely knowledgeable about Amazon India policies.

Your role is to help customers with orders, payments, deliveries, returns, refunds, EMI,
Amazon Pay, UPI, Prime membership, damaged products, and account issues.

Always:
- Greet the customer warmly and acknowledge their problem with empathy
- Provide clear, step-by-step instructions
- Reference specific Amazon features like Your Orders, Your Account, Manage Prime
- Mention relevant policies like A-to-Z Guarantee, return windows, refund timelines
- End with an offer to help with anything else

Amazon India specific facts:
- Customer care: 1800-1200-1637 toll free 24/7
- UPI only works on Amazon India app, not browser
- Most items have 10-day return window
- Refunds: Credit or Debit card 3-5 days, UPI 2-4 days, Amazon Pay instant
- EMI available on Credit cards, Debit cards, Amazon Pay Later, Bajaj Finserv
- No Cost EMI means interest is pre-adjusted in price so total equals product price
- A-to-Z Guarantee covers third-party seller orders for delivery and condition
"""

CUSTOMER_SYSTEM_PROMPT = """You are a real Amazon India customer with a specific problem.
Generate ONE realistic customer message about your issue.
- Write naturally like a real Indian customer texting support
- Show appropriate emotion such as frustrated, confused, worried, or polite
- Be specific but do not always give all details upfront
- Keep it 1-4 sentences
Return ONLY the customer message, nothing else."""


class OllamaBackend:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.base_url = "http://localhost:11434"

    def is_available(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if not any(self.model.split(":")[0] in m for m in models):
                    print(f"  [Ollama] Model '{self.model}' not found. Available: {models}")
                    print(f"  [Ollama] Pull it with: ollama pull {self.model}")
                    return False
                return True
        except Exception:
            pass
        return False

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {"temperature": temperature}
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class GroqBackend:
    def __init__(self, model: str = "llama3-70b-8192"):
        self.model = model
        self.api_key = os.environ.get("GROQ_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": temperature,
            "max_tokens": 600
        }
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload, timeout=30
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class TemplateBackend:
    def is_available(self) -> bool:
        return True

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# CONVERSATION GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class ConversationGenerator:
    def __init__(self, backend, mode: str = "template"):
        self.backend = backend
        self.mode = mode

    def _customer_msg(self, scenario: dict) -> str:
        variation = random.choice(scenario["variations"])
        if self.mode == "template":
            return variation
        try:
            prompt = (
                f"The customer's issue is about: {scenario['topic'].replace('_', ' ')}\n"
                f"Base their message on this example but rephrase it: \"{variation}\"\n"
                f"Write a realistic customer support message."
            )
            return self.backend.chat(CUSTOMER_SYSTEM_PROMPT, prompt, temperature=0.85)
        except Exception as e:
            print(f"    [LLM customer fallback] {e}")
            return variation

    def _expert_resp(self, scenario: dict, customer_msg: str) -> str:
        if self.mode == "template":
            return EXPERT_ANSWERS.get(scenario["topic"], EXPERT_ANSWERS["general"])
        try:
            prompt = (
                f"Customer message: \"{customer_msg}\"\n\n"
                f"Topic: {scenario['topic'].replace('_', ' ')}\n\n"
                f"Provide a complete, helpful Amazon India support response."
            )
            return self.backend.chat(AMAZON_SYSTEM_PROMPT, prompt, temperature=0.6)
        except Exception as e:
            print(f"    [LLM expert fallback] {e}")
            return EXPERT_ANSWERS.get(scenario["topic"], EXPERT_ANSWERS["general"])

    def generate(self, num_turns: int = None) -> str:
        scenario = random.choice(CUSTOMER_SCENARIOS)
        if num_turns is None:
            num_turns = random.randint(2, 5)

        lines = [
            f"[Amazon India Customer Support Conversation]",
            f"[Topic: {scenario['topic'].replace('_', ' ').title()}]",
            "",
        ]

        # Turn 1: opening
        c_msg = self._customer_msg(scenario)
        lines += [f"Customer: {c_msg}", ""]
        lines += [f"Support Agent Priya: {self._expert_resp(scenario, c_msg)}", ""]

        # Follow-up turns
        for turn in range(num_turns - 1):
            fq = random.choice(FOLLOWUP_QUESTIONS).format(
                order_id=random.randint(1000000, 9999999),
                days=random.randint(2, 10),
            )

            if self.mode != "template":
                try:
                    fq = self.backend.chat(
                        CUSTOMER_SYSTEM_PROMPT,
                        f"Customer wants to follow up after receiving support. Base it on: \"{fq}\"",
                        temperature=0.9,
                    )
                except Exception:
                    pass

            lines += [f"Customer: {fq}", ""]

            if turn == num_turns - 2:
                agent_resp = random.choice(CLOSING_RESPONSES)
            else:
                agent_resp = random.choice(INTERMEDIATE_RESPONSES)

            lines += [f"Support Agent Priya: {agent_resp}", ""]

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def detect_backend(mode: str, ollama_model: str):
    if mode == "template":
        print("  [Mode] Template (built-in, zero dependency)")
        return TemplateBackend(), "template"

    if mode in ("auto", "ollama"):
        b = OllamaBackend(model=ollama_model)
        if b.is_available():
            print(f"  [Mode] Ollama (local LLM: {ollama_model})")
            return b, "ollama"
        elif mode == "ollama":
            print(f"  [ERROR] Ollama not available.")
            print(f"          Start it with: ollama serve")
            print(f"          Pull model with: ollama pull {ollama_model}")
            sys.exit(1)

    if mode in ("auto", "groq"):
        b = GroqBackend()
        if b.is_available():
            print("  [Mode] Groq API (llama3-70b, free tier)")
            return b, "groq"
        elif mode == "groq":
            print("  [ERROR] GROQ_API_KEY not set.")
            print("          Get a free key at https://console.groq.com")
            print("          Then: export GROQ_API_KEY=your_key_here")
            sys.exit(1)

    print("  [Mode] Template (fallback — Ollama not running, GROQ_API_KEY not set)")
    print("  Tip: For richer conversations, run: ollama serve && ollama pull llama3.2")
    return TemplateBackend(), "template"


def save_conv(text: str, incoming_dir: str, idx: int) -> str:
    os.makedirs(incoming_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(incoming_dir, f"amazon_conv_{ts}_{idx:04d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run_training(ckpt_dir: str, incoming_dir: str, steps: int, lr: float):
    import subprocess
    print("\n  Training model on new conversations...")
    result = subprocess.run(
        [sys.executable, "continual_learn.py",
         "--ckpt_dir", ckpt_dir,
         "--incoming_dir", incoming_dir,
         "--steps_per_doc", str(steps),
         "--lr", str(lr)],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"    {line}")
    if result.returncode != 0 and result.stderr:
        print(f"    [Warning] {result.stderr.strip()[:300]}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Amazon India support conversations for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
        Quick Start:
          python3 self_play_trainer.py                      # generate 20 conversations (template mode)
          python3 self_play_trainer.py --num 100            # generate 100 conversations
          python3 self_play_trainer.py --watch --train      # continuous generate + train loop

        With Ollama (free, local LLM — best quality):
          1. Install Ollama: https://ollama.com
          2. ollama serve
          3. ollama pull llama3.2
          4. python3 self_play_trainer.py --mode ollama --watch --train

        With Groq (free API, high quality):
          1. Get free key: https://console.groq.com
          2. export GROQ_API_KEY=your_key
          3. python3 self_play_trainer.py --mode groq --watch --train
        """)
    )
    parser.add_argument("--num", type=int, default=20,
                        help="Conversations to generate per batch (default: 20)")
    parser.add_argument("--mode", choices=["auto", "ollama", "groq", "template"], default="auto",
                        help="LLM backend to use (default: auto)")
    parser.add_argument("--ollama_model", default="llama3.2",
                        help="Ollama model name (default: llama3.2)")
    parser.add_argument("--incoming_dir", default="incoming",
                        help="Directory to save conversations (default: incoming/)")
    parser.add_argument("--watch", action="store_true",
                        help="Run continuously, generating new batches forever")
    parser.add_argument("--train", action="store_true",
                        help="Auto-run continual_learn.py after each batch")
    parser.add_argument("--ckpt_dir", default="checkpoints",
                        help="Checkpoint directory for training (default: checkpoints/)")
    parser.add_argument("--steps_per_doc", type=int, default=30,
                        help="Training steps per conversation file (default: 30)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for fine-tuning (default: 1e-4)")
    parser.add_argument("--batch_interval", type=int, default=60,
                        help="Seconds between batches in watch mode (default: 60)")
    parser.add_argument("--turns", type=int, default=None,
                        help="Turns per conversation (default: random 2-5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each generated conversation to screen")
    args = parser.parse_args()

    backend, mode = detect_backend(args.mode, args.ollama_model)
    gen = ConversationGenerator(backend=backend, mode=mode)

    w = 70
    print("\n" + "=" * w)
    print("  Amazon India Support — Self-Play Training Pipeline")
    print(f"  Generating {args.num} conversations per batch in [{mode.upper()}] mode")
    print("=" * w)

    total = 0
    batch = 0

    while True:
        batch += 1
        print(f"\n  Batch {batch} — generating {args.num} conversations...")
        print("  " + "-" * 65)

        for i in range(args.num):
            try:
                conv = gen.generate(num_turns=args.turns)
                path = save_conv(conv, args.incoming_dir, total + i)
                topic = conv.split("[Topic: ")[1].split("]")[0] if "[Topic: " in conv else "?"
                print(f"  [{i+1:3d}/{args.num}] OK  {topic:40s}  {os.path.basename(path)}")

                if args.verbose:
                    print("\n" + "-" * 60)
                    print(conv[:600] + ("..." if len(conv) > 600 else ""))
                    print("-" * 60 + "\n")

            except Exception as e:
                print(f"  [{i+1:3d}/{args.num}] ERR {e}")

        total += args.num
        print(f"\n  Batch {batch} done. Total conversations: {total}")
        print(f"  Saved to: {os.path.abspath(args.incoming_dir)}/")

        if args.train:
            if os.path.exists(os.path.join(args.ckpt_dir, "model.pt")):
                run_training(args.ckpt_dir, args.incoming_dir, args.steps_per_doc, args.lr)
            else:
                print(f"\n  No checkpoint at '{args.ckpt_dir}/'. Run train.py first.")

        if not args.watch:
            break

        print(f"\n  Waiting {args.batch_interval}s before next batch (Ctrl+C to stop)...")
        try:
            time.sleep(args.batch_interval)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break

    print(f"\n{'=' * w}")
    print(f"  Self-play complete! Generated {total} conversations")
    print(f"  Location: {os.path.abspath(args.incoming_dir)}/")
    if not args.train:
        print(f"\n  To train your model:")
        print(f"    python3 continual_learn.py --incoming_dir {args.incoming_dir}")
    print(f"{'=' * w}\n")


if __name__ == "__main__":
    main()
