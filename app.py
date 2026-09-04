"""
app.py — Flask backend for Amazon India Support AI Web UI
"""

import re
import sys
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

# ── Import support logic from support_chat.py ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from support_chat import SupportExecutive, classify_intent, KNOWLEDGE_BASE

app = Flask(__name__)

# ── One shared session per server process (for demo; extend with sessions) ──
agent = SupportExecutive()

# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE  (served at GET /)
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Amazon India — Customer Support</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
/* ── CSS RESET & TOKENS ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --amz-orange:   #FF9900;
  --amz-orange-d: #E47911;
  --amz-dark:     #131921;
  --amz-navy:     #232F3E;
  --amz-navy2:    #37475A;
  --amz-teal:     #007185;
  --amz-teal-h:   #C7F0F5;
  --amz-yellow:   #FFF3CD;
  --bg:           #EAEDED;
  --card:         #FFFFFF;
  --text:         #0F1111;
  --text-muted:   #565959;
  --border:       #D5D9D9;
  --agent-bubble: #F0F2F2;
  --user-bubble:  #232F3E;
  --shadow:       0 2px 5px rgba(213,217,217,.5);
  --shadow-lg:    0 4px 24px rgba(0,0,0,.14);
  --radius:       10px;
  --radius-sm:    6px;
  --font:         'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --transition:   0.22s cubic-bezier(.4,0,.2,1);
}

html, body { height: 100%; }
body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  font-size: 14px;
}

/* ── AMAZON HEADER ──────────────────────────────────────────────── */
.amz-header {
  background: var(--amz-dark);
  padding: 0 20px;
  height: 60px;
  display: flex;
  align-items: center;
  gap: 20px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 8px rgba(0,0,0,.4);
}

.amz-logo {
  display: flex;
  align-items: center;
  gap: 2px;
  text-decoration: none;
  border: 1.5px solid transparent;
  padding: 4px 8px;
  border-radius: 3px;
  transition: border-color var(--transition);
}
.amz-logo:hover { border-color: white; }
.amz-logo-text {
  font-size: 22px;
  font-weight: 700;
  color: white;
  letter-spacing: -0.5px;
}
.amz-logo-text span { color: var(--amz-orange); }
.amz-logo-dot {
  width: 8px; height: 8px;
  background: var(--amz-orange);
  border-radius: 50%;
  margin-bottom: -8px;
  margin-left: 1px;
}

.amz-nav-text {
  color: #ccc;
  font-size: 11px;
  line-height: 1.2;
}
.amz-nav-text strong { color: white; font-size: 13px; display: block; }

.amz-search {
  flex: 1;
  max-width: 620px;
  display: flex;
  height: 40px;
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 0 0 0 3px var(--amz-orange);
}
.amz-search input {
  flex: 1;
  padding: 0 12px;
  border: none;
  outline: none;
  font-size: 14px;
  font-family: var(--font);
  background: white;
}
.amz-search-btn {
  width: 46px;
  background: var(--amz-orange);
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background var(--transition);
}
.amz-search-btn:hover { background: var(--amz-orange-d); }
.amz-search-btn svg { fill: #333; width: 18px; height: 18px; }

.amz-header-actions { display: flex; gap: 8px; margin-left: auto; }
.amz-header-btn {
  color: white;
  font-size: 12px;
  text-align: center;
  padding: 4px 10px;
  border: 1.5px solid transparent;
  border-radius: 3px;
  cursor: pointer;
  white-space: nowrap;
  transition: border-color var(--transition);
  background: none;
  font-family: var(--font);
  line-height: 1.3;
}
.amz-header-btn span { font-size: 13px; font-weight: 600; display: block; }
.amz-header-btn:hover { border-color: white; }

/* ── SUB-NAV ────────────────────────────────────────────────────── */
.amz-subnav {
  background: var(--amz-navy);
  height: 38px;
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 4px;
  overflow-x: auto;
}
.amz-subnav::-webkit-scrollbar { display: none; }
.subnav-item {
  color: white;
  font-size: 13px;
  padding: 6px 12px;
  border-radius: 3px;
  white-space: nowrap;
  cursor: pointer;
  border: 1.5px solid transparent;
  transition: border-color var(--transition);
  font-weight: 500;
}
.subnav-item:hover { border-color: white; }
.subnav-item.active { border-color: white; }

/* ── MAIN LAYOUT ────────────────────────────────────────────────── */
.main-wrap {
  flex: 1;
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 0;
  max-width: 1280px;
  margin: 24px auto;
  width: 100%;
  padding: 0 16px;
  height: calc(100vh - 140px);
}

/* ── SIDEBAR ────────────────────────────────────────────────────── */
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-right: 16px;
}

.sidebar-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 16px;
  box-shadow: var(--shadow);
}
.sidebar-card h3 {
  font-size: 13px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

.quick-btn {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 9px 12px;
  margin-bottom: 6px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--card);
  cursor: pointer;
  font-size: 12.5px;
  color: var(--amz-teal);
  font-weight: 500;
  text-align: left;
  transition: all var(--transition);
  font-family: var(--font);
}
.quick-btn:last-child { margin-bottom: 0; }
.quick-btn:hover {
  background: var(--amz-teal-h);
  border-color: var(--amz-teal);
  transform: translateX(2px);
}
.quick-btn .icon {
  font-size: 16px;
  flex-shrink: 0;
  width: 22px;
  text-align: center;
}

.status-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: #067D62;
  flex-shrink: 0;
  animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
  0%,100% { box-shadow: 0 0 0 0 rgba(6,125,98,.4); }
  50% { box-shadow: 0 0 0 4px rgba(6,125,98,0); }
}

/* ── CHAT PANEL ─────────────────────────────────────────────────── */
.chat-panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-lg);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Chat header */
.chat-header {
  background: linear-gradient(135deg, var(--amz-dark) 0%, var(--amz-navy) 100%);
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.agent-avatar {
  width: 46px; height: 46px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--amz-orange) 0%, #FF6B35 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  font-weight: 700;
  color: white;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(255,153,0,.4);
}
.agent-info { flex: 1; }
.agent-name { color: white; font-weight: 700; font-size: 15px; }
.agent-role { color: #aaa; font-size: 12px; margin-top: 2px; }
.agent-status {
  display: flex;
  align-items: center;
  gap: 5px;
  color: #4CAF50;
  font-size: 11.5px;
  margin-top: 3px;
  font-weight: 500;
}
.agent-status::before {
  content: '';
  width: 6px; height: 6px;
  border-radius: 50%;
  background: #4CAF50;
}
.chat-header-actions { display: flex; gap: 8px; }
.header-icon-btn {
  width: 32px; height: 32px;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,.2);
  background: rgba(255,255,255,.08);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  transition: background var(--transition);
}
.header-icon-btn:hover { background: rgba(255,255,255,.18); }

/* Chat messages */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  scroll-behavior: smooth;
  background: #FAFAFA;
}
.chat-messages::-webkit-scrollbar { width: 4px; }
.chat-messages::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* Message rows */
.msg-row {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  animation: fadeSlideIn 0.3s ease;
}
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user { flex-direction: row-reverse; }

.msg-avatar {
  width: 32px; height: 32px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 600;
}
.msg-avatar.agent-av {
  background: linear-gradient(135deg, var(--amz-orange), #FF6B35);
  color: white;
}
.msg-avatar.user-av {
  background: var(--amz-navy);
  color: white;
}

.msg-content { max-width: 68%; display: flex; flex-direction: column; gap: 4px; }
.msg-row.user .msg-content { align-items: flex-end; }

.bubble {
  padding: 11px 14px;
  border-radius: 16px;
  line-height: 1.55;
  font-size: 13.5px;
  word-break: break-word;
  white-space: pre-wrap;
}
.bubble.agent {
  background: var(--agent-bubble);
  color: var(--text);
  border-bottom-left-radius: 4px;
  border: 1px solid var(--border);
}
.bubble.user {
  background: var(--amz-navy);
  color: white;
  border-bottom-right-radius: 4px;
}

/* Markdown-like formatting inside bubbles */
.bubble strong { font-weight: 700; }
.bubble .step-num {
  display: inline-block;
  width: 20px; height: 20px;
  background: var(--amz-orange);
  color: white;
  border-radius: 50%;
  text-align: center;
  line-height: 20px;
  font-size: 11px;
  font-weight: 700;
  margin-right: 6px;
  flex-shrink: 0;
}
.bubble .bullet-line {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  margin: 3px 0;
}
.bubble .bullet-icon { flex-shrink: 0; font-size: 13px; margin-top: 1px; }

.msg-time {
  font-size: 10.5px;
  color: var(--text-muted);
  padding: 0 4px;
}

/* Typing indicator */
.typing-row { display: flex; gap: 10px; align-items: flex-end; }
.typing-bubble {
  background: var(--agent-bubble);
  border: 1px solid var(--border);
  border-radius: 16px;
  border-bottom-left-radius: 4px;
  padding: 12px 16px;
  display: flex;
  gap: 5px;
  align-items: center;
}
.typing-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #999;
  animation: typingBounce 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.18s; }
.typing-dot:nth-child(3) { animation-delay: 0.36s; }
@keyframes typingBounce {
  0%,60%,100% { transform: translateY(0); background: #bbb; }
  30% { transform: translateY(-6px); background: var(--amz-orange); }
}

/* Suggested chips */
.suggestions-bar {
  padding: 8px 20px 4px;
  display: flex;
  gap: 8px;
  overflow-x: auto;
  background: #FAFAFA;
}
.suggestions-bar::-webkit-scrollbar { display: none; }
.suggestion-chip {
  flex-shrink: 0;
  padding: 6px 13px;
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 12px;
  color: var(--amz-teal);
  background: white;
  cursor: pointer;
  white-space: nowrap;
  transition: all var(--transition);
  font-family: var(--font);
  font-weight: 500;
}
.suggestion-chip:hover {
  background: var(--amz-teal-h);
  border-color: var(--amz-teal);
  transform: translateY(-1px);
}

/* Chat input */
.chat-input-area {
  padding: 14px 16px;
  border-top: 1px solid var(--border);
  background: white;
  display: flex;
  gap: 10px;
  align-items: flex-end;
}
.chat-input-wrap {
  flex: 1;
  display: flex;
  align-items: center;
  background: #F8F8F8;
  border: 1.5px solid var(--border);
  border-radius: 24px;
  padding: 8px 16px;
  gap: 8px;
  transition: border-color var(--transition), box-shadow var(--transition);
}
.chat-input-wrap:focus-within {
  border-color: var(--amz-teal);
  box-shadow: 0 0 0 3px rgba(0,113,133,.12);
  background: white;
}
#chat-input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  font-size: 13.5px;
  font-family: var(--font);
  color: var(--text);
  resize: none;
  max-height: 100px;
  line-height: 1.4;
}
#chat-input::placeholder { color: #999; }

.send-btn {
  width: 42px; height: 42px;
  border-radius: 50%;
  border: none;
  background: var(--amz-orange);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
  transition: all var(--transition);
  box-shadow: 0 2px 8px rgba(255,153,0,.35);
}
.send-btn:hover {
  background: var(--amz-orange-d);
  transform: scale(1.08);
  box-shadow: 0 4px 14px rgba(255,153,0,.4);
}
.send-btn:active { transform: scale(0.96); }
.send-btn:disabled { background: #ddd; box-shadow: none; cursor: not-allowed; transform: none; }

/* ── AMAZON FOOTER ──────────────────────────────────────────────── */
.amz-footer-top {
  background: var(--amz-navy2);
  text-align: center;
  padding: 12px;
  color: white;
  font-size: 12px;
  cursor: pointer;
  transition: background var(--transition);
}
.amz-footer-top:hover { background: #4a5568; }
.amz-footer-bottom {
  background: var(--amz-navy);
  display: flex;
  justify-content: center;
  gap: 16px;
  padding: 10px 20px;
  flex-wrap: wrap;
}
.footer-link { color: #DDD; font-size: 11.5px; text-decoration: none; transition: color var(--transition); }
.footer-link:hover { color: white; }
.amz-footer-copy {
  background: var(--amz-dark);
  text-align: center;
  padding: 8px;
  color: #777;
  font-size: 11px;
}

/* ── TOAST ──────────────────────────────────────────────────────── */
.toast {
  position: fixed;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%) translateY(20px);
  background: var(--amz-navy);
  color: white;
  padding: 10px 20px;
  border-radius: 20px;
  font-size: 13px;
  opacity: 0;
  transition: all 0.3s ease;
  pointer-events: none;
  z-index: 999;
}
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

/* ── RESPONSIVE ─────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .main-wrap { grid-template-columns: 1fr; height: calc(100vh - 120px); }
  .sidebar { display: none; }
  .amz-search { display: none; }
}
</style>
</head>
<body>

<!-- ── AMAZON HEADER ─────────────────────────────────────────── -->
<header class="amz-header">
  <a class="amz-logo" href="#">
    <span class="amz-logo-text">amazon<span>.in</span></span>
  </a>

  <div class="amz-nav-text" style="display:none">
    <strong>Deliver to</strong>India
  </div>

  <div class="amz-search">
    <input type="text" placeholder="Search Amazon.in" id="search-box"/>
    <button class="amz-search-btn" title="Search">
      <svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27A6.5 6.5 0 0 0 16 9.5 6.5 6.5 0 0 0 9.5 3 6.5 6.5 0 0 0 3 9.5 6.5 6.5 0 0 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
    </button>
  </div>

  <div class="amz-header-actions">
    <button class="amz-header-btn">Hello, Rahul<span>Account &amp; Lists ▾</span></button>
    <button class="amz-header-btn">Returns<span>&amp; Orders</span></button>
    <button class="amz-header-btn" style="font-size:22px;font-weight:700;letter-spacing:-1px">🛒<span style="font-size:11px">Cart</span></button>
  </div>
</header>

<!-- ── SUB-NAV ───────────────────────────────────────────────── -->
<nav class="amz-subnav">
  <div class="subnav-item active">≡ All</div>
  <div class="subnav-item">Today's Deals</div>
  <div class="subnav-item">Customer Service</div>
  <div class="subnav-item">Prime</div>
  <div class="subnav-item">New Releases</div>
  <div class="subnav-item">Electronics</div>
  <div class="subnav-item">Fashion</div>
  <div class="subnav-item">Books</div>
  <div class="subnav-item">Groceries</div>
  <div class="subnav-item">Home &amp; Kitchen</div>
</nav>

<!-- ── MAIN LAYOUT ───────────────────────────────────────────── -->
<main class="main-wrap">

  <!-- ── SIDEBAR ─────────────────────────────────────────────── -->
  <aside class="sidebar">
    <div class="sidebar-card">
      <h3>Quick Solutions</h3>
      <button class="quick-btn" onclick="sendQuick('Where is my order?')">
        <span class="icon">📦</span>Your Orders — Track or cancel
      </button>
      <button class="quick-btn" onclick="sendQuick('How do I return a product?')">
        <span class="icon">🔄</span>Returns &amp; Refunds
      </button>
      <button class="quick-btn" onclick="sendQuick('How do I manage Prime?')">
        <span class="icon">⭐</span>Manage Prime
      </button>
      <button class="quick-btn" onclick="sendQuick('My payment failed on Amazon')">
        <span class="icon">💳</span>Payment Settings
      </button>
      <button class="quick-btn" onclick="sendQuick('I see an unknown charge on my account')">
        <span class="icon">🎁</span>Account &amp; Charges
      </button>
    </div>

    <div class="sidebar-card">
      <h3>Support Status</h3>
      <div class="status-row">
        <div class="status-dot"></div>
        Priya is online and ready
      </div>
      <div class="status-row" style="gap:8px;margin-top:6px">
        <span>⏱</span><span>Avg. response: &lt;5 sec</span>
      </div>
      <div class="status-row" style="gap:8px;margin-top:4px">
        <span>📞</span><span>Call: 1800-1200-1637</span>
      </div>
    </div>

    <div class="sidebar-card">
      <h3>Browse Help Topics</h3>
      <button class="quick-btn" onclick="sendQuick('My delivery is late')">
        <span class="icon">🚚</span>Delivery Issues
      </button>
      <button class="quick-btn" onclick="sendQuick('I received a damaged product')">
        <span class="icon">📋</span>Damaged / Defective
      </button>
      <button class="quick-btn" onclick="sendQuick('How do I pay in EMI?')">
        <span class="icon">💰</span>EMI &amp; Offers
      </button>
      <button class="quick-btn" onclick="sendQuick('My UPI payment failed')">
        <span class="icon">📱</span>UPI / PayLater
      </button>
    </div>
  </aside>

  <!-- ── CHAT PANEL ───────────────────────────────────────────── -->
  <section class="chat-panel">

    <!-- Chat header -->
    <div class="chat-header">
      <div class="agent-avatar">P</div>
      <div class="agent-info">
        <div class="agent-name">Priya</div>
        <div class="agent-role">Amazon India · Customer Support Executive</div>
        <div class="agent-status">Active now</div>
      </div>
      <div class="chat-header-actions">
        <button class="header-icon-btn" title="Rate this chat" onclick="showToast('Thanks for your feedback! 😊')">★</button>
        <button class="header-icon-btn" title="Clear chat" onclick="clearChat()">↺</button>
      </div>
    </div>

    <!-- Suggested chips -->
    <div class="suggestions-bar" id="suggestions">
      <div class="suggestion-chip" onclick="sendQuick('Where is my order?')">📦 Track order</div>
      <div class="suggestion-chip" onclick="sendQuick('My payment failed on Amazon')">💳 Payment issue</div>
      <div class="suggestion-chip" onclick="sendQuick('How do I return a product?')">🔄 Return item</div>
      <div class="suggestion-chip" onclick="sendQuick('My delivery is late')">🚚 Late delivery</div>
      <div class="suggestion-chip" onclick="sendQuick('I received a damaged product')">📋 Damaged item</div>
      <div class="suggestion-chip" onclick="sendQuick('How do I cancel my order?')">❌ Cancel order</div>
      <div class="suggestion-chip" onclick="sendQuick('My UPI payment failed')">📱 UPI failed</div>
      <div class="suggestion-chip" onclick="sendQuick('How do I pay in EMI?')">💰 EMI options</div>
    </div>

    <!-- Messages -->
    <div class="chat-messages" id="messages"></div>

    <!-- Input -->
    <div class="chat-input-area">
      <div class="chat-input-wrap">
        <textarea id="chat-input" rows="1" placeholder="Type your message here…"
          onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      </div>
      <button class="send-btn" id="send-btn" onclick="sendMessage()" title="Send">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
        </svg>
      </button>
    </div>
  </section>
</main>

<!-- ── FOOTER ────────────────────────────────────────────────── -->
<div class="amz-footer-top" onclick="window.scrollTo({top:0,behavior:'smooth'})">
  Back to top
</div>
<footer class="amz-footer-bottom">
  <a class="footer-link" href="#">Conditions of Use</a>
  <a class="footer-link" href="#">Privacy Notice</a>
  <a class="footer-link" href="#">Interest-Based Ads</a>
  <a class="footer-link" href="#">© 2024, Amazon.com, Inc. or its affiliates</a>
</footer>
<div class="amz-footer-copy">
  © 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
</div>

<div class="toast" id="toast"></div>

<!-- ── JAVASCRIPT ────────────────────────────────────────────── -->
<script>
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('chat-input');
const sendBtn    = document.getElementById('send-btn');

let isThinking = false;

// ── Render helpers ────────────────────────────────────────────
function formatText(text) {
  // Bold **text**
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Numbered steps "  1. ..." or "Step 1:"
  text = text.replace(/^(\s*)(Step\s+)?(\d+)[\.:\)]\s+/gm, (_, sp, st, n) =>
    `${sp}<span class="step-num">${n}</span>`);
  // Bullet lines starting with emoji or dash
  text = text.replace(/^(\s*)([-•▸►✅❌💳📦🌐📍🔢⏳🔄🚫📊🇺🇸💰🆔⏱🏦📱📮🌧️📋💥🗺️📞🔒🚪🐕📵🏠✔️ℹ️⚠️💡🎁⭐📅🔗👨‍👩‍👧])\s+/gm,
    (_, sp, bullet) => `${sp}<span class="bullet-line"><span class="bullet-icon">${bullet}</span>`
  );
  // Inline | table-like formatting → keep as-is (pre-wrap handles it)
  return text;
}

function nowTime() {
  return new Date().toLocaleTimeString('en-IN', {hour:'2-digit', minute:'2-digit'});
}

function appendMessage(text, role, time) {
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const av = document.createElement('div');
  av.className = `msg-avatar ${role === 'user' ? 'user-av' : 'agent-av'}`;
  av.textContent = role === 'user' ? '👤' : 'P';

  const content = document.createElement('div');
  content.className = 'msg-content';

  const bubble = document.createElement('div');
  bubble.className = `bubble ${role === 'user' ? 'user' : 'agent'}`;
  bubble.innerHTML = role === 'user' ? escHtml(text) : formatText(escHtml(text));

  const timeEl = document.createElement('div');
  timeEl.className = 'msg-time';
  timeEl.textContent = time || nowTime();

  content.appendChild(bubble);
  content.appendChild(timeEl);

  if (role === 'user') {
    row.appendChild(content);
    row.appendChild(av);
  } else {
    row.appendChild(av);
    row.appendChild(content);
  }

  messagesEl.appendChild(row);
  scrollBottom();
  return row;
}

function escHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function showTyping() {
  const row = document.createElement('div');
  row.className = 'typing-row';
  row.id = 'typing-indicator';

  const av = document.createElement('div');
  av.className = 'msg-avatar agent-av';
  av.textContent = 'P';

  const bubble = document.createElement('div');
  bubble.className = 'typing-bubble';
  bubble.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

  row.appendChild(av);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollBottom();
}

function removeTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function scrollBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
}

// ── Send message ──────────────────────────────────────────────
async function sendMessage(overrideText) {
  const text = (overrideText || inputEl.value).trim();
  if (!text || isThinking) return;

  isThinking = true;
  sendBtn.disabled = true;
  inputEl.value = '';
  autoResize(inputEl);

  // Hide suggestions after first message
  document.getElementById('suggestions').style.display = 'none';

  appendMessage(text, 'user');
  showTyping();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    removeTyping();
    appendMessage(data.response, 'agent');
  } catch (e) {
    removeTyping();
    appendMessage('Sorry, I\'m having trouble connecting. Please try again or call us at 1800-1200-1637.', 'agent');
  }

  isThinking = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

function sendQuick(text) { sendMessage(text); }

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function clearChat() {
  messagesEl.innerHTML = '';
  document.getElementById('suggestions').style.display = 'flex';
  fetch('/api/reset', { method: 'POST' });
  // Re-trigger greeting
  setTimeout(() => fetchGreeting(), 300);
  showToast('Chat cleared ↺');
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2800);
}

// ── Initial greeting ──────────────────────────────────────────
async function fetchGreeting() {
  showTyping();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: 'hello' })
    });
    const data = await res.json();
    removeTyping();
    appendMessage(data.response, 'agent');
  } catch(e) {
    removeTyping();
    appendMessage('Hello! I\'m Priya from Amazon India Customer Support. How can I help you today?', 'agent');
  }
}

// ── Boot ──────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  inputEl.focus();
  setTimeout(fetchGreeting, 600);
});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "Please type a message."}), 400

    response = agent.respond(message)
    return jsonify({
        "response":  response,
        "intent":    classify_intent(message),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    global agent
    agent = SupportExecutive()
    return jsonify({"status": "ok"})


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status":        "online",
        "agent":         agent.AGENT_NAME,
        "turns":         agent.turn_count,
        "customer_name": agent.customer_name,
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default=None,
                        help="Path to GPT checkpoint dir for model-assisted fallback")
    args = parser.parse_args()

    if args.model:
        try:
            import torch
            from model import GPT
            from tokenizer import CharTokenizer
            ckpt = torch.load(f"{args.model}/model.pt", map_location="cpu")
            tok  = CharTokenizer.load(f"{args.model}/vocab.json")
            mdl  = GPT(**ckpt["config"])
            mdl.load_state_dict(ckpt["model_state"])
            mdl.eval()
            agent.model     = mdl
            agent.tokenizer = tok
            agent.config    = ckpt["config"]
            print(f"✅ Loaded GPT model from '{args.model}'")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}. Running rule-based only.")

    print(f"\n{'='*60}")
    print(f"  🛒  Amazon India Support AI — Web UI")
    print(f"  Open: http://{args.host}:{args.port}")
    print(f"{'='*60}\n")
    app.run(host=args.host, port=args.port, debug=False)
