"""
E-Commerce Strategy AI Team — Live Dashboard
=============================================
Reads agent_events.jsonl written by main.py and renders live agent status.

Run:
  streamlit run dashboard.py
"""

import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

EVENTS_FILE = Path("C:/ClaudeAI/agent_events.jsonl")

ALL_AGENTS = [
    "research-expert",
    "analyst-expert",
    "price-intelligence-agent",
    "channel-agent",
    "packaging-kitting-agent",
    "content-listing-agent",
    "strategist-expert",
]

AGENT_PHASES = {
    "research-expert":          1,
    "analyst-expert":           1,
    "price-intelligence-agent": 2,
    "channel-agent":            2,
    "packaging-kitting-agent":  3,
    "content-listing-agent":    3,
    "strategist-expert":        4,
}

PHASE_NAMES = {
    0: "Not started",
    1: "Phase 1 — Foundation Research",
    2: "Phase 2 — Commercial Intelligence",
    3: "Phase 3 — Offer and Content",
    4: "Phase 4 — Strategy Synthesis",
}

STATUS_ICON = {
    "waiting": "⬜",
    "running": "🔵",
    "done":    "✅",
    "failed":  "❌",
}

STATUS_COLOR = {
    "waiting": "#aaa",
    "running": "#1a73e8",
    "done":    "#0a8f4f",
    "failed":  "#c62828",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_events() -> list:
    if not EVENTS_FILE.exists():
        return []
    events = []
    with open(EVENTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def build_state(events: list) -> dict:
    state = {
        "topic":          None,
        "current_phase":  0,
        "agent_status":   {a: "waiting" for a in ALL_AGENTS},
        "agent_message":  {a: "—"       for a in ALL_AGENTS},
        "report_path":    None,
        "run_complete":   False,
        "run_error":      None,
    }

    for e in events:
        t     = e.get("type", "")
        agent = e.get("agent", "")

        if t == "run_start":
            state["topic"] = e.get("topic")

        elif t == "agent_started":
            if agent in state["agent_status"]:
                state["agent_status"][agent]  = "running"
                state["agent_message"][agent] = "Starting…"
                phase = AGENT_PHASES.get(agent, 0)
                if phase > state["current_phase"]:
                    state["current_phase"] = phase

        elif t == "agent_progress":
            if agent in state["agent_status"]:
                turns  = e.get("turns",  "?")
                tokens = e.get("tokens", "?")
                state["agent_message"][agent] = f"Turn {turns} · ~{tokens} tokens"

        elif t == "agent_done":
            if agent in state["agent_status"]:
                state["agent_status"][agent]  = "done"
                state["agent_message"][agent] = "Completed ✓"

        elif t == "agent_failed":
            if agent in state["agent_status"]:
                state["agent_status"][agent]  = "failed"
                state["agent_message"][agent] = e.get("error", "Failed")

        elif t == "run_complete":
            state["run_complete"] = True
            state["report_path"]  = e.get("report_path")

        elif t == "run_error":
            state["run_error"] = e.get("error")

    return state

# ── Rendering helpers ─────────────────────────────────────────────────────────

def agent_card(name: str, status: str, message: str) -> None:
    icon  = STATUS_ICON[status]
    color = STATUS_COLOR[status]
    phase = AGENT_PHASES.get(name, 0)
    # Shorten display name: strip "-expert" and "-agent" suffixes
    label = name.replace("-expert", "").replace("-agent", "")
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e0e0e0;
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 6px;
        ">
            <div style="font-weight: 700; font-size: 0.88em;">{icon}&nbsp;{label}</div>
            <div style="font-size: 0.72em; color: #888; margin-top: 2px;">
                {name} &nbsp;·&nbsp; Phase {phase}
            </div>
            <div style="font-size: 0.8em; color: {color}; margin-top: 6px;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_event(e: dict) -> str:
    raw_ts = e.get("ts", "")
    # Extract HH:MM:SS from ISO timestamp
    ts    = raw_ts[11:19] if len(raw_ts) >= 19 else raw_ts
    etype = e.get("type", "?")
    agent = e.get("agent", "")

    if etype == "run_start":
        msg = f"**Run started** — {e.get('topic', '')}"
    elif etype == "agent_started":
        msg = f"**{agent}** started (Phase {e.get('phase', '?')})"
    elif etype == "agent_progress":
        msg = f"**{agent}** — turn {e.get('turns','?')} · {e.get('tokens','?')} tokens"
    elif etype == "agent_done":
        msg = f"**{agent}** completed ✓"
    elif etype == "agent_failed":
        msg = f"**{agent}** FAILED — {e.get('error','')}"
    elif etype == "team_lead_msg":
        text = e.get("text", "")
        snippet = text[:100] + ("…" if len(text) > 100 else "")
        msg = f"Team Lead: _{snippet}_"
    elif etype == "run_complete":
        msg = f"**Run complete** — {e.get('report_path', '')}"
    elif etype == "run_error":
        msg = f"**ERROR** — {e.get('error', '')}"
    else:
        msg = etype

    return f"`{ts}` &nbsp; {msg}"

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="E-Commerce Strategy AI — Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────

events = load_events()
state  = build_state(events)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📊 E-Commerce Strategy AI Team")
st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}   ·   Auto-refresh every 1 s")
st.divider()

# ── No data state ─────────────────────────────────────────────────────────────

if not events:
    st.info("No active run detected. Start **main.py** to see live updates.")
    st.markdown(
        "```\npython main.py \"your product topic\"\n```"
    )

# ── Active / completed run ────────────────────────────────────────────────────

else:
    # Run metadata
    col_topic, col_phase = st.columns(2)
    with col_topic:
        st.metric("Topic", state["topic"] or "—")
    with col_phase:
        st.metric("Current Phase", PHASE_NAMES.get(state["current_phase"], "—"))

    # Completion / error banners
    if state["run_complete"]:
        st.success(f"✅ **Run complete.** Report saved to: `{state['report_path']}`")
    elif state["run_error"]:
        st.error(f"❌ **Run error:** {state['run_error']}")

    st.divider()

    # ── Agent cards — 3 columns ───────────────────────────────────────────────

    st.subheader("Agent Status")

    # Row 1: Phase 1 + first Phase 2 agent
    r1c1, r1c2, r1c3 = st.columns(3)
    # Row 2: second Phase 2 + Phase 3 agents
    r2c1, r2c2, r2c3 = st.columns(3)
    # Row 3: Phase 4 agent (strategist spans first column)
    r3c1, r3c2, r3c3 = st.columns(3)

    grid = [
        (r1c1, "research-expert"),
        (r1c2, "analyst-expert"),
        (r1c3, "price-intelligence-agent"),
        (r2c1, "channel-agent"),
        (r2c2, "packaging-kitting-agent"),
        (r2c3, "content-listing-agent"),
        (r3c1, "strategist-expert"),
    ]

    for col, agent in grid:
        with col:
            agent_card(
                agent,
                state["agent_status"][agent],
                state["agent_message"][agent],
            )

    st.divider()

    # ── Event log ─────────────────────────────────────────────────────────────

    st.subheader("Event Log")

    # Newest first, cap at 80 lines
    lines = [format_event(e) for e in reversed(events[-80:])]
    st.markdown("\n\n".join(lines) if lines else "_No events yet._")

# ── Auto-refresh every 1 second ───────────────────────────────────────────────

time.sleep(1)
st.rerun()
