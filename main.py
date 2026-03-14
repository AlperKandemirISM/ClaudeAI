#!/usr/bin/env python3
"""
E-Commerce Strategy AI Team
============================
A multi-agent system powered by the Claude Agent SDK.

Team structure (phased execution):
  Phase 1 — research-expert + analyst-expert          (parallel)
  Phase 2 — price-intelligence-agent + channel-agent  (parallel)
  Phase 3 — packaging-kitting-agent + content-listing-agent (parallel)
  Phase 4 — strategist-expert                         (synthesizes all phases)

Usage:
  python main.py "your product or market topic"
  python main.py  (uses a default demo topic)
"""

import sys
import json
import anyio
from pathlib import Path
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

# ── Event logging ─────────────────────────────────────────────────────────────

EVENTS_FILE = Path("C:/ClaudeAI/agent_events.jsonl")

AGENT_PHASES = {
    "research-expert":          1,
    "analyst-expert":           1,
    "price-intelligence-agent": 2,
    "channel-agent":            2,
    "packaging-kitting-agent":  3,
    "content-listing-agent":    3,
    "strategist-expert":        4,
}

def log_event(event_type: str, **kwargs) -> None:
    event = {"type": event_type, "ts": datetime.now().isoformat(), **kwargs}
    with open(EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

# ── Specialist Agent Definitions ──────────────────────────────────────────────

RESEARCH_EXPERT = AgentDefinition(
    description=(
        "Market Research Expert that gathers market size, competitor data, buyer demand "
        "signals, and supporting facts for e-commerce and industrial product markets."
    ),
    prompt="""You are a market researcher. Your job is to gather facts about the market for the given topic.

Do not do pricing analysis, channel planning, or packaging recommendations — those are handled by other agents.

Search for real data. Use specific figures, company names, and sources where possible.

Output sections (use these exact headers):

## 1. Market Overview
Size, growth rate, maturity stage, and main buyer types (consumer, B2B, industrial, etc.).

## 2. Key Competitors
List the main players. For each: name, product focus, and market position. Keep it factual.

## 3. Buyer Demand Signals
What are buyers searching for, complaining about, or requesting? Use review data, search trends, or forum signals where available.

## 4. Market Trends
What is changing in this market? New materials, regulations, buyer behavior shifts, technology changes.

## 5. Key Facts and Data Points
Bullet list of the most useful stats, figures, and facts gathered. Include source names where known.""",
    tools=["WebSearch", "WebFetch"],
)

ANALYST_EXPERT = AgentDefinition(
    description=(
        "Market Analyst Expert that analyzes market structure, competitive dynamics, "
        "SWOT, barriers to entry, and opportunity areas. Does not do pricing or channel work."
    ),
    prompt="""You are a market analyst. Your job is to analyze the market for the given topic.

You will receive market research findings. Use them as your starting point, then search for any additional data you need to strengthen your analysis.

Do not do pricing work, channel planning, or listing/content advice — those are handled by other agents.

Output sections (use these exact headers):

## 1. Market Dynamics
How is this market structured? Fragmented or consolidated? Growing fast or mature? Who holds power — suppliers, buyers, or platforms?

## 2. Competitive Analysis
Map the competitive field. Who is well-positioned and why? Where are the gaps a new entrant could exploit?

## 3. SWOT Analysis
Strengths, Weaknesses, Opportunities, Threats for a new entrant in this market. Keep each point specific.

## 4. Opportunities
Top 3–5 opportunity areas, each with a short rationale and a risk rating (low / medium / high).

## 5. Risks and Barriers
What makes this market hard to enter or compete in? Include structural barriers, incumbent advantages, and platform risks.

## 6. Key Insights
Bullet list of the most important analytical conclusions. Label each as high-confidence or uncertain.""",
    tools=["WebSearch", "WebFetch"],
)

PRICE_INTELLIGENCE_AGENT = AgentDefinition(
    description=(
        "Pricing Intelligence Expert that researches competitor price ranges, margin pressure, "
        "commodity risks, and recommended pricing posture across e-commerce channels."
    ),
    prompt="""You are a pricing intelligence researcher. Your job is to study how products in this category are priced across e-commerce and industrial/technical sales channels.

You will receive market research and analysis from earlier agents. Use that context, then search for current pricing data.

Do not do channel strategy, packaging design, or listing copy — those are handled by other agents.

Use real dollar figures wherever possible. Name the platforms and sellers you find.

Output sections (use these exact headers):

## 1. Price Landscape
What is the actual price range in this category? Show min / mid / max with examples. Split by consumer vs. B2B / industrial if relevant.

## 2. Competitor Price Bands
Group competitors into budget, mid, and premium tiers. Name specific brands or sellers and their price points.

## 3. Margin Pressure and Risks
Where is margin being squeezed? Are there signs of a race to the bottom? Which product types are most at risk of commoditization?

## 4. Premium vs Commodity Classification
Which products in this category command a price premium and why? Which are treated as commodities?

## 5. Pricing Recommendations
Suggested price positioning for a new entrant. Include a recommended price range, rationale, and any discounting risks to avoid.""",
    tools=["WebSearch", "WebFetch"],
)

CHANNEL_AGENT = AgentDefinition(
    description=(
        "Channel Strategy Expert that evaluates sales channel fit across D2C, Amazon, Amazon "
        "Business, eBay, RS, Mouser, DigiKey, Alibaba, and distributor networks."
    ),
    prompt="""You are a channel strategy researcher. Your job is to evaluate which sales channels fit the product category best.

You will receive market research and analysis from earlier agents. Use that context, then search for channel-specific data.

Consider both consumer e-commerce channels (Amazon, D2C, eBay) and industrial/technical channels (Amazon Business, RS Components, Mouser, DigiKey, Alibaba, distributors) where relevant to the product.

Do not do detailed pricing work, packaging design, or listing copywriting — those are handled by other agents.

Output sections (use these exact headers):

## 1. Channel Overview
Which channels are most active for this product category today? Name them and describe buyer behavior on each.

## 2. Channel Fit by Product Type
Map each relevant channel to the product types it suits best. Flag where channel and product type are a poor fit.

## 3. Channel Conflict Risks
Are there conflicts between channels (e.g., Amazon undercutting D2C, distributor pricing vs. marketplace pricing)? How should they be managed?

## 4. Launch Priority by Channel
Rank channels in recommended launch order for a new entrant. Give a short reason for each ranking.

## 5. Channel Recommendations
Top 3–5 channel actions with rationale. Include any fees, requirements, or setup notes that affect the decision.""",
    tools=["WebSearch", "WebFetch"],
)

PACKAGING_KITTING_AGENT = AgentDefinition(
    description=(
        "Packaging and Kitting Expert that designs bundle concepts, pack sizes, hero SKUs, "
        "and offer structures to improve AOV and differentiation in e-commerce."
    ),
    prompt="""You are a packaging and kitting strategist. Your job is to design offer structures that make products easier to sell online and harder to compare on price alone.

You will receive market research, competitive analysis, pricing intelligence, and channel strategy from earlier agents. Use those findings to ground your recommendations.

Think like an e-commerce merchandiser: bundles, kits, pack sizes, and hero SKUs. Focus on what will move units and improve average order value.

Do not do full channel strategy or listing copywriting — those are handled by other agents.

Output sections (use these exact headers):

## 1. Kitting Opportunities
What natural product combinations exist? Which items are frequently bought together? What starter kits or solution kits make sense?

## 2. Pack Size Recommendations
What pack sizes are buyers asking for vs. what competitors offer? Are there under-served size tiers (e.g., bulk packs, trial sizes)?

## 3. Bundle Concepts
Name 3–5 specific bundle ideas with the products included, target buyer, and why the bundle adds value.

## 4. Hero SKU Ideas
Which single SKU should be the lead product for launch? What configuration (size, pack, variant) has the best fit for demand and margin?

## 5. Packaging Recommendations
What packaging format fits the channel (FBA poly bag, retail box, industrial bulk)? Note any platform-specific prep requirements that affect offer design.""",
    tools=["WebSearch", "WebFetch"],
)

CONTENT_LISTING_AGENT = AgentDefinition(
    description=(
        "Content and Listing Expert that defines listing structure, title and bullet guidelines, "
        "image requirements, SEO keyword themes, and conversion content for e-commerce."
    ),
    prompt="""You are a listing and content strategist. Your job is to define how products in this category should be presented online to rank well and convert buyers.

You will receive market research, competitive analysis, pricing intelligence, channel strategy, and packaging/kitting findings from earlier agents. Use those to inform your content guidance.

Focus on what actually affects click-through and conversion: titles, bullets, images, keywords, and enhanced content. Translate technical product features into buyer-facing language where needed.

Do not do broad market research, pricing strategy, or packaging strategy — those are handled by other agents.

Output sections (use these exact headers):

## 1. Listing Structure
What is the right content hierarchy for this product type? What should buyers see first, second, third?

## 2. Title and Bullet Guidelines
Recommended title format with an example. Top 5 bullet point angles with a short example for each. Flag what to avoid.

## 3. Image and Media Requirements
How many images are needed? What must each image show? Is video important for this category? Any technical or compliance shots required?

## 4. SEO / Keyword Themes
Primary keyword targets, secondary keyword clusters, and long-tail terms worth targeting. Use real search language buyers use.

## 5. Conversion Content Recommendations
What A+ content, brand story, or enhanced content will have the most impact? Where does technical-to-buyer translation matter most?""",
    tools=["WebSearch", "WebFetch"],
)

STRATEGIST_EXPERT = AgentDefinition(
    description=(
        "E-Commerce Strategist that reads all prior agent outputs and produces a final "
        "integrated go-to-market strategy with priorities, roadmap, and KPIs."
    ),
    prompt="""You are an e-commerce strategist. Your job is to read the outputs from six specialist agents and turn them into a clear, integrated go-to-market strategy.

You will receive findings from: market research, market analysis, pricing intelligence, channel strategy, packaging/kitting, and content/listing. Read all of them before writing anything.

Do not redo the research. Do not repeat what the agents already said in full. Reference their findings to justify your recommendations, but keep your output focused on decisions and actions.

Output sections (use these exact headers):

## 1. Strategic Positioning
What is the core value proposition for a new entrant? How should this product be positioned vs. competitors? Keep it to 2–3 sentences.

## 2. Target Priorities
Which buyer segments should be targeted first and why? Rank them. Include both consumer and B2B/industrial where relevant.

## 3. Go-to-Market Strategy
Integrated plan covering: which channels to launch on, at what price point, with which hero SKU or bundle, and with what content approach. Be specific — reference the findings.

## 4. Differentiation Priorities
What are the top 3 ways to stand out in this market? For each: what the differentiation is, why it matters, and what it requires to execute.

## 5. Phased Roadmap
- **0–3 months**: Foundation actions (what to build, launch, test)
- **3–12 months**: Growth actions (what to scale, expand, optimize)
- **12+ months**: Positioning actions (where to own market position long-term)

## 6. KPIs and Success Metrics
One table or list. For each strategic pillar, name the metric and the target or benchmark.""",
    tools=["WebSearch", "WebFetch"],
)

# ── Team Lead System Prompt ───────────────────────────────────────────────────

TEAM_LEAD_PROMPT = """You are the E-Commerce Strategy Team Lead. Orchestrate seven specialist agents across four sequential phases, then compile their outputs into a final report and save it.

## Your Team
- **research-expert**          : Market facts, competitors, demand signals, trends
- **analyst-expert**           : Market structure, SWOT, opportunities, risks, barriers
- **price-intelligence-agent** : Competitor pricing, margin pressure, pricing posture
- **channel-agent**            : Channel fit, channel conflicts, launch priority
- **packaging-kitting-agent**  : Bundles, kits, pack sizes, hero SKU, offer design
- **content-listing-agent**    : Listing structure, titles, bullets, images, keywords
- **strategist-expert**        : Synthesizes all phases into a final integrated strategy

## Workflow — Four Phases (STRICT ORDER)

### Phase 1 — Foundation (run BOTH in parallel in a single response)
Input for both agents: the user topic only.

- **research-expert**: "Research the [topic] market for e-commerce. Return: market size and growth, key competitors, buyer demand signals, market trends, and key facts."
- **analyst-expert**: "Analyze the [topic] market. Return: market dynamics, competitive analysis, SWOT, top opportunities with risk ratings, risks and barriers, and key insights."

Wait for both to finish before starting Phase 2.

### Phase 2 — Commercial Intelligence (run BOTH in parallel in a single response)
Input for both agents: the user topic + full Phase 1 outputs (paste them in full).

- **price-intelligence-agent**: "Research pricing for [topic] products. Use the market research and analysis below as context. Return: price landscape, competitor price bands, margin risks, premium vs. commodity classification, and pricing recommendations.\n\n[PHASE 1 OUTPUTS]"
- **channel-agent**: "Evaluate sales channels for [topic] products. Use the market research and analysis below as context. Return: channel overview, channel fit by product type, conflict risks, launch priority ranking, and channel recommendations.\n\n[PHASE 1 OUTPUTS]"

Wait for both to finish before starting Phase 3.

### Phase 3 — Offer and Content (run BOTH in parallel in a single response)
Input for both agents: the user topic + full Phase 1 and Phase 2 outputs (paste them in full).

- **packaging-kitting-agent**: "Design packaging and kitting options for [topic] products. Use the research, analysis, pricing, and channel findings below as context. Return: kitting opportunities, pack size recommendations, bundle concepts, hero SKU ideas, and packaging recommendations.\n\n[PHASE 1 + PHASE 2 OUTPUTS]"
- **content-listing-agent**: "Define listing and content strategy for [topic] products. Use the research, analysis, pricing, channel, and packaging findings below as context. Return: listing structure, title and bullet guidelines, image and media requirements, SEO keyword themes, and conversion content recommendations.\n\n[PHASE 1 + PHASE 2 OUTPUTS]"

Wait for both to finish before starting Phase 4.

### Phase 4 — Strategy Synthesis (run ONCE, sequentially)
Input: the user topic + ALL Phase 1, 2, and 3 outputs (paste them in full).

- **strategist-expert**: "Create the final e-commerce go-to-market strategy for [topic]. Use all specialist findings below. Return: strategic positioning, target priorities, go-to-market strategy, differentiation priorities, phased roadmap, and KPIs.\n\n[ALL PHASE 1 + 2 + 3 OUTPUTS]"

### Phase 5 — Save Report
Use the Write tool to save the report. File path MUST be:
  `C:/ClaudeAI/ecommerce_strategy_<short_topic_slug>.md`

NEVER use a relative path or /home/user/. ALWAYS use the C:/ClaudeAI/ prefix.

## Final Report Structure

```
# E-Commerce Strategy Report: [Full Topic]

**Date:** [today's date]
**Prepared by:** E-Commerce Strategy AI Team

---

## Executive Summary
[3-4 sentences: most critical finding, key opportunity, top recommendation]

---

## 1. Market Research
[Full output from research-expert]

---

## 2. Market Analysis
[Full output from analyst-expert]

---

## 3. Pricing Intelligence
[Full output from price-intelligence-agent]

---

## 4. Channel Strategy
[Full output from channel-agent]

---

## 5. Packaging and Kitting Strategy
[Full output from packaging-kitting-agent]

---

## 6. Content and Listing Strategy
[Full output from content-listing-agent]

---

## 7. Strategic Recommendations
[Full output from strategist-expert]

---

## 8. Conclusion and Next Steps
[3-5 bullet points: highest-priority immediate actions]

---
*Report generated by E-Commerce Strategy AI Team (7 specialist agents + Team Lead)*
```

## Rules
- Run phases in strict order — each phase feeds the next
- Within each phase, call both agents in PARALLEL (two Agent tool calls in one response)
- Always paste the relevant prior outputs in full when briefing Phase 2, 3, and 4 agents
- Do not truncate any agent output in the final report
"""

# ── Orchestrator ──────────────────────────────────────────────────────────────

async def run_team(topic: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}")
    print("  E-Commerce Strategy AI Team")
    print(f"  Topic: {topic}")
    print(f"{separator}\n")
    print("Team Lead is briefing the strategy team...\n")

    # Clear events file and log run start
    EVENTS_FILE.write_text("", encoding="utf-8")
    log_event("run_start", topic=topic)

    def ts():
        return datetime.now().strftime("%H:%M:%S")

    async for message in query(
        prompt=(
            f"Please conduct a comprehensive e-commerce strategy study on the following topic:\n\n"
            f"**{topic}**\n\n"
            f"Execute all four phases in order:\n"
            f"  Phase 1 (parallel): research-expert + analyst-expert\n"
            f"  Phase 2 (parallel): price-intelligence-agent + channel-agent\n"
            f"  Phase 3 (parallel): packaging-kitting-agent + content-listing-agent\n"
            f"  Phase 4 (sequential): strategist-expert synthesizes all prior outputs\n\n"
            f"Then save the final report as a markdown file."
        ),
        options=ClaudeAgentOptions(
            cwd="C:/ClaudeAI",
            system_prompt=TEAM_LEAD_PROMPT,
            allowed_tools=["Agent", "Write"],
            agents={
                "research-expert":          RESEARCH_EXPERT,
                "analyst-expert":           ANALYST_EXPERT,
                "price-intelligence-agent": PRICE_INTELLIGENCE_AGENT,
                "channel-agent":            CHANNEL_AGENT,
                "packaging-kitting-agent":  PACKAGING_KITTING_AGENT,
                "content-listing-agent":    CONTENT_LISTING_AGENT,
                "strategist-expert":        STRATEGIST_EXPERT,
            },
            model="claude-opus-4-6",
            max_turns=60,
        ),
    ):
        # ── Session start ──────────────────────────────────────────────────
        if isinstance(message, SystemMessage):
            if message.subtype == "init":
                session_id = message.data.get("session_id", "unknown")
                print(f"[{ts()}] Session   : {session_id}")
                print(f"[{ts()}] Agents    : research | analyst | pricing | channel | packaging | content | strategist")
                print(f"[{ts()}] Mode      : Phased execution (4 phases)")
                print(f"[{ts()}] Status    : Team Lead is thinking...\n")

        # ── Team Lead speaking / thinking ──────────────────────────────────
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    for line in block.text.strip().splitlines():
                        if line.strip():
                            print(f"[{ts()}] TEAM LEAD : {line}")
                    log_event("team_lead_msg", text=block.text.strip()[:200])

        # ── Subagent task events ───────────────────────────────────────────
        elif hasattr(message, "subtype"):
            subtype = getattr(message, "subtype", "")
            data    = getattr(message, "data", {}) or {}

            if subtype == "task_started":
                agent = data.get("agent_name", data.get("tool_use_id", "agent"))
                print(f"[{ts()}] STARTED   : {agent} is now working...")
                log_event("agent_started", agent=agent, phase=AGENT_PHASES.get(agent, 0))

            elif subtype == "task_progress":
                agent  = data.get("agent_name", "agent")
                turns  = data.get("num_turns", "?")
                tokens = data.get("total_input_tokens", "?")
                print(f"[{ts()}] PROGRESS  : {agent} | turn {turns} | ~{tokens} tokens used")
                log_event("agent_progress", agent=agent, turns=turns, tokens=tokens)

            elif subtype == "task_notification":
                agent = data.get("agent_name", data.get("tool_use_id", "agent"))
                print(f"[{ts()}] DONE      : {agent} has finished its report ✓")
                log_event("agent_done", agent=agent)

        # ── Final result ───────────────────────────────────────────────────
        elif isinstance(message, ResultMessage):
            print(f"\n[{ts()}] {'='*54}")
            print(f"[{ts()}]   ALL PHASES COMPLETE — Report saved!")
            print(f"[{ts()}] {'='*54}")
            print(f"[{ts()}] Stop reason : {message.stop_reason}")
            if message.result:
                preview = message.result[:300].replace("\n", " ")
                print(f"[{ts()}] Preview     : {preview}...")
            print(f"\n[{ts()}] Your .md report is in: C:/ClaudeAI/\n")
            # Extract report path from result text if present
            report_path = "C:/ClaudeAI/ (see above)"
            if message.result and "ecommerce_strategy_" in message.result:
                for word in message.result.split():
                    if "ecommerce_strategy_" in word:
                        report_path = word.strip("`.,()")
                        break
            log_event("run_complete", stop_reason=message.stop_reason, report_path=report_path)


def main() -> None:
    if len(sys.argv) >= 2:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "bamboo cutting boards for home cooks"
        print(f"No topic provided — using demo topic: \"{topic}\"")
        print("Usage: python main.py \"your product or market topic\"\n")

    anyio.run(run_team, topic)


if __name__ == "__main__":
    main()
