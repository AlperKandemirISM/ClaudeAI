#!/usr/bin/env python3
"""
Market Research AI Team
=======================
A multi-agent system powered by the Claude Agent SDK.

Team structure:
  - Team Lead      : Orchestrates all agents and writes the final report
  - Research Expert: Gathers market data, trends, and competitive intelligence
  - Analyst Expert : Analyzes patterns, risks, and opportunities
  - Strategist     : Develops actionable go-to-market strategies

Usage:
  python main.py "your market research topic"
  python main.py  (uses a default demo topic)
"""

import sys
import anyio
from datetime import datetime
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AgentDefinition,
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    TextBlock,
)

# ── Specialist Agent Definitions ──────────────────────────────────────────────

RESEARCH_EXPERT = AgentDefinition(
    description=(
        "Market Research Expert that gathers comprehensive market data, current "
        "trends, competitor intelligence, and consumer insights for any given topic."
    ),
    prompt="""You are a Market Research Expert with deep expertise in gathering market intelligence.

Your responsibilities:
- Research current market size, growth rate, and trajectory
- Identify key players, direct/indirect competitors, and their positioning
- Gather consumer behavior patterns, pain points, and demand signals
- Analyze market segments and target audiences
- Spot emerging trends, disruptive forces, and technology shifts
- Collect supporting statistics, data points, and real-world examples

Structure your output with clear sections:
1. Market Overview (size, growth, maturity)
2. Key Players & Competitive Landscape
3. Consumer Insights & Demand Signals
4. Emerging Trends & Disruptors
5. Key Data Points & Sources

Be thorough, cite specific figures where found, and highlight the most important findings.""",
    tools=["WebSearch", "WebFetch"],
)

ANALYST_EXPERT = AgentDefinition(
    description=(
        "Market Analyst Expert that performs rigorous analysis on market data to "
        "uncover patterns, competitive positioning gaps, risks, and growth opportunities."
    ),
    prompt="""You are a Market Analyst Expert with strong quantitative and qualitative analysis skills.

Your responsibilities:
- Analyze market data to uncover patterns, cycles, and inflection points
- Map the competitive landscape and identify white-space opportunities
- Conduct SWOT analysis of the market and leading players
- Evaluate market opportunities with risk/reward assessment
- Identify barriers to entry, moats, and critical success factors
- Quantify addressable market segments and growth potential
- Surface the most actionable insights from the available data

Structure your output with clear sections:
1. Market Dynamics & Pattern Analysis
2. Competitive Landscape & Positioning Map
3. SWOT Analysis
4. Opportunity Assessment (with risk ratings)
5. Barriers to Entry & Success Factors
6. Key Analytical Insights

Support every claim with reasoning. Flag high-confidence vs. uncertain conclusions.""",
    tools=["WebSearch", "WebFetch"],
)

STRATEGIST_EXPERT = AgentDefinition(
    description=(
        "Market Strategist Expert that develops concrete, prioritized go-to-market "
        "strategies and actionable recommendations based on market intelligence."
    ),
    prompt="""You are a Market Strategist Expert with experience designing winning market strategies.

Your responsibilities:
- Develop concrete, actionable go-to-market strategies
- Define target segments, value propositions, and positioning
- Identify differentiation and competitive advantage opportunities
- Prioritize strategic initiatives with effort/impact mapping
- Recommend partnership, channel, and ecosystem strategies
- Propose a phased roadmap with milestones
- Define KPIs and success metrics for each strategic pillar

Structure your output with clear sections:
1. Strategic Positioning & Value Proposition
2. Target Segment Priorities
3. Go-to-Market Strategy (channels, messaging, partnerships)
4. Competitive Differentiation Playbook
5. Phased Strategic Roadmap (0-3 months, 3-12 months, 12+ months)
6. KPIs & Success Metrics

Make every recommendation specific and implementable — avoid generic advice.""",
    tools=["WebSearch", "WebFetch"],
)

# ── Team Lead System Prompt ───────────────────────────────────────────────────

TEAM_LEAD_PROMPT = """You are the Market Research Team Lead. Your job is to orchestrate three specialist agents in PARALLEL, then synthesize their outputs into a polished executive-ready report.

## Your Team
- **research-expert** : Gathers market data, trends, and competitive intelligence
- **analyst-expert**  : Analyzes patterns, risks, opportunities, and competitive positioning
- **strategist-expert**: Develops actionable go-to-market strategies and KPIs

## Workflow

### Step 1 — Parallel Dispatch (do this in ONE response)
Call the Agent tool THREE TIMES simultaneously, one per specialist. Give each a focused task prompt tailored to the research topic. Do NOT wait for one before calling the next.

Example task prompts:
- research-expert : "Research the [topic] market. Cover: market size/growth, key players, consumer insights, emerging trends. Provide data-backed findings."
- analyst-expert  : "Analyze the [topic] market. Cover: competitive dynamics, SWOT, opportunity/risk assessment, success factors. Be analytical and specific."
- strategist-expert: "Develop a go-to-market strategy for [topic]. Cover: positioning, target segments, GTM channels, differentiation, phased roadmap, and KPIs."

### Step 2 — Synthesize
Once all three agents respond, combine their outputs into the final report structure below.

### Step 3 — Save Report
Use the Write tool to save the report. The file_path MUST be the full absolute path:
  `C:/ClaudeAI/market_research_<short_topic_slug>.md`
(e.g., topic "AI SaaS tools 2025" → `C:/ClaudeAI/market_research_ai_saas_tools_2025.md`)

NEVER save to /home/user/ or any relative path. ALWAYS use the C:/ClaudeAI/ prefix.

## Final Report Structure

```
# Market Research Report: [Full Topic]

**Date:** [today's date]
**Prepared by:** Market Research AI Team

---

## Executive Summary
[3-4 sentences covering the most critical findings, key opportunity, and top strategic recommendation]

---

## 1. Market Research Findings
[Full output from research-expert, lightly edited for flow]

---

## 2. Market Analysis
[Full output from analyst-expert, lightly edited for flow]

---

## 3. Strategic Recommendations
[Full output from strategist-expert, lightly edited for flow]

---

## 4. Conclusion & Next Steps
[2-3 bullet points summarising the most important actions to take]

---
*Report generated by Market Research AI Team (Team Lead + Research Expert + Analyst Expert + Strategist Expert)*
```

## Critical Rules
- ALWAYS spawn all three agents in PARALLEL in a single response (three Agent tool calls at once)
- Preserve the full detail from each specialist — do not truncate their outputs
- The saved .md file is the deliverable; make it comprehensive and executive-ready
"""

# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_team(topic: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}")
    print("  Market Research AI Team")
    print(f"  Topic: {topic}")
    print(f"{separator}\n")
    print("Team Lead is briefing the research team...\n")

    def ts():
        return datetime.now().strftime("%H:%M:%S")

    async for message in query(
        prompt=(
            f"Please conduct a comprehensive market research study on the following topic:\n\n"
            f"**{topic}**\n\n"
            f"Spawn all three specialist agents in parallel now, then synthesize their outputs "
            f"into the final report and save it as a markdown file."
        ),
        options=ClaudeAgentOptions(
            cwd="C:/ClaudeAI",
            system_prompt=TEAM_LEAD_PROMPT,
            allowed_tools=["Agent", "Write"],
            agents={
                "research-expert": RESEARCH_EXPERT,
                "analyst-expert": ANALYST_EXPERT,
                "strategist-expert": STRATEGIST_EXPERT,
            },
            model="claude-opus-4-6",
            max_turns=40,
        ),
    ):
        # ── Session start ──────────────────────────────────────────────────
        if isinstance(message, SystemMessage):
            if message.subtype == "init":
                session_id = message.data.get("session_id", "unknown")
                print(f"[{ts()}] Session   : {session_id}")
                print(f"[{ts()}] Agents    : research-expert | analyst-expert | strategist-expert")
                print(f"[{ts()}] Mode      : Parallel execution")
                print(f"[{ts()}] Status    : Team Lead is thinking...\n")

        # ── Team Lead speaking / thinking ──────────────────────────────────
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    # Print each line with a timestamp prefix
                    for line in block.text.strip().splitlines():
                        if line.strip():
                            print(f"[{ts()}] TEAM LEAD : {line}")

        # ── Subagent task events ───────────────────────────────────────────
        elif hasattr(message, "subtype"):
            subtype = getattr(message, "subtype", "")
            data    = getattr(message, "data", {}) or {}

            if subtype == "task_started":
                agent = data.get("agent_name", data.get("tool_use_id", "agent"))
                print(f"[{ts()}] STARTED   : {agent} is now working...")

            elif subtype == "task_progress":
                agent   = data.get("agent_name", "agent")
                turns   = data.get("num_turns", "?")
                tokens  = data.get("total_input_tokens", "?")
                print(f"[{ts()}] PROGRESS  : {agent} | turn {turns} | ~{tokens} tokens used")

            elif subtype == "task_notification":
                agent = data.get("agent_name", data.get("tool_use_id", "agent"))
                print(f"[{ts()}] DONE      : {agent} has finished its report ✓")

        # ── Final result ───────────────────────────────────────────────────
        elif isinstance(message, ResultMessage):
            print(f"\n[{ts()}] {'='*54}")
            print(f"[{ts()}]   ALL AGENTS COMPLETE — Report saved!")
            print(f"[{ts()}] {'='*54}")
            print(f"[{ts()}] Stop reason : {message.stop_reason}")
            if message.result:
                preview = message.result[:300].replace("\n", " ")
                print(f"[{ts()}] Preview     : {preview}...")
            print(f"\n[{ts()}] Your .md report is in: C:/ClaudeAI/\n")


def main() -> None:
    if len(sys.argv) >= 2:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "AI-powered SaaS tools for small businesses in 2025"
        print(f"No topic provided — using demo topic: \"{topic}\"")
        print("Usage: python main.py \"your market research topic\"\n")

    anyio.run(run_team, topic)


if __name__ == "__main__":
    main()
