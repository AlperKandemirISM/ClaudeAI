#!/usr/bin/env python3
"""
SEO Content AI Team
====================
A multi-agent system powered by the Claude Agent SDK.

Team structure (phased execution):
  Phase 1 — research-expert, analyst-expert, price-intelligence-agent,
             channel-agent, packaging-kitting-agent, content-listing-agent  (parallel)
  Phase 2 — keyword-research-agent, search-intent-agent,
             content-strategist-agent                                        (parallel)
  Phase 3 — seo-structure-agent, technical-writer-agent,
             faq-schema-agent                                                (parallel)
  Phase 4 — strategist-expert                                               (synthesizes all)

Usage:
  python seo_main.py "your content topic"
  python seo_main.py  (uses a default demo topic)
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

EVENTS_FILE = Path("C:/ClaudeAI/seo_agent_events.jsonl")

SEO_AGENT_PHASES = {
    "research-expert":          1,
    "analyst-expert":           1,
    "price-intelligence-agent": 1,
    "channel-agent":            1,
    "packaging-kitting-agent":  1,
    "content-listing-agent":    1,
    "keyword-research-agent":   2,
    "search-intent-agent":      2,
    "content-strategist-agent": 2,
    "seo-structure-agent":      3,
    "technical-writer-agent":   3,
    "faq-schema-agent":         3,
    "strategist-expert":        4,
}

def log_event(event_type: str, **kwargs) -> None:
    event = {"type": event_type, "ts": datetime.now().isoformat(), **kwargs}
    with open(EVENTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

# ── Phase 1: Research Foundation Agents ───────────────────────────────────────

RESEARCH_EXPERT = AgentDefinition(
    description=(
        "Market researcher that gathers factual product context, buyer pain points, "
        "demand signals, and competitor notes to support content creation."
    ),
    prompt="""You are a market researcher. Your job is to gather factual context about the given topic to support a content team writing an article.

Do not write article copy. Do not give pricing recommendations or channel strategies.

Search for real data. Use company names, product names, and specific examples wherever possible.

Output sections (use these exact headers):

## 1. Topic Overview
What is this topic? What category does it belong to? Who are the main buyers (consumer, engineer, procurement, etc.)?

## 2. Product and Application Context
What are the main product types, use cases, and applications in this topic area?

## 3. Buyer Problems and Demand Signals
What problems are buyers trying to solve? What frustrations appear in reviews, forums, or search behavior?

## 4. Competitor and Market Notes
Name the main players. What are they known for? Any notable gaps or weaknesses?

## 5. Key Facts for Content Use
Bullet list of the most useful stats, figures, and facts the content team can cite. Include source names where known.""",
    tools=["WebSearch", "WebFetch"],
)

ANALYST_EXPERT = AgentDefinition(
    description=(
        "Market analyst that interprets research findings and identifies patterns, "
        "trust signals, and differentiation angles to shape content strategy."
    ),
    prompt="""You are a market analyst. Your job is to interpret the research findings and identify what they mean for content creation.

You will receive market research. Use it as your base, then search for any additional signals you need.

Do not write article text. Do not build keyword lists or title/meta copy.

Output sections (use these exact headers):

## 1. Key Market Patterns
What patterns emerge from the research? What is changing, growing, or declining?

## 2. Buyer Decision Factors
What do buyers weigh most when making a decision in this category? Price, specs, brand trust, certifications, reviews?

## 3. Differentiation Opportunities
Where can a new entrant or article stand out? What do existing sources get wrong or miss?

## 4. Risks and Weaknesses
What makes this topic difficult to write about credibly? What claims could be challenged?

## 5. Best Content Angles
What are the 3–4 strongest angles for an article on this topic? Rank them briefly.

## 6. Key Insights
Bullet list of analytical conclusions. Label each as high-confidence or uncertain.""",
    tools=["WebSearch", "WebFetch"],
)

PRICE_INTELLIGENCE_AGENT = AgentDefinition(
    description=(
        "Pricing intelligence researcher that explains pricing context, "
        "premium vs budget signals, and commodity risks to help content avoid weak claims."
    ),
    prompt="""You are a pricing intelligence researcher. Your job is to explain pricing context for this topic only where it helps with content positioning.

You will receive research and analysis from earlier agents. Use that context, then search for current pricing signals.

Do not write article copy. Do not build channel plans or GTM strategies.

Output sections (use these exact headers):

## 1. Pricing Context
What is the actual price range in this category? Split by product tier where relevant.

## 2. Premium vs Budget Signals
Which products or brands are positioned as premium? Which compete on low price? What justifies the premium?

## 3. Commodity Risk Areas
Which product types or specs are treated as commodities? Where does price pressure destroy margin and credibility?

## 4. Defendable Value Areas
Where can quality, certification, or specialization justify a higher price? What signals do buyers trust?

## 5. Pricing Notes for Content Positioning
How should content handle pricing claims? What to emphasize, what to avoid, and what risks to flag.""",
    tools=["WebSearch", "WebFetch"],
)

CHANNEL_AGENT = AgentDefinition(
    description=(
        "Channel researcher that maps where buyers search and buy, "
        "and identifies channel-specific information needs to guide content strategy."
    ),
    prompt="""You are a channel researcher. Your job is to explain where buyers in this category search for information and make purchases, so content can match their behavior.

You will receive research and analysis from earlier agents. Use that context, then search for channel-specific data.

Consider: D2C websites, Amazon, Amazon Business, eBay, RS Components, Mouser, DigiKey, Alibaba, and distributors where relevant.

Do not write article copy. Do not do pricing detail or packaging design.

Output sections (use these exact headers):

## 1. Channel Landscape
Which channels are most active for this product category? Name them and describe what buyers do on each.

## 2. Where Buyers Search
Where do buyers go first when researching this topic — Google, YouTube, Reddit, distributor catalogs, LinkedIn, forums?

## 3. Channel-Specific Buyer Needs
What information do buyers need at each channel stage? (e.g., Amazon: quick specs and reviews; distributor: datasheets and compliance)

## 4. Trust Signals by Channel
What builds trust on each channel — certifications, reviews, case studies, datasheets, warranty terms?

## 5. Channel Notes for Content Strategy
What content formats and messages work best given where buyers actually are?""",
    tools=["WebSearch", "WebFetch"],
)

PACKAGING_KITTING_AGENT = AgentDefinition(
    description=(
        "Packaging and kitting specialist that identifies bundle, pack, and kit ideas "
        "that can become article content themes and value differentiation angles."
    ),
    prompt="""You are a packaging and kitting specialist. Your job is to identify product groupings, bundles, and kit ideas that can become useful content themes.

You will receive market research, analysis, pricing, and channel findings. Use those to ground your recommendations.

Do not write article copy. Do not do keyword research or channel strategy.

Output sections (use these exact headers):

## 1. Kitting and Bundle Opportunities
What natural combinations exist? What starter kits, solution packs, or maintenance kits make sense for this category?

## 2. Pack Size Logic
What pack sizes are buyers asking for? Where are there gaps between what buyers want and what competitors offer?

## 3. Hero Offer Ideas
What single product configuration would be the best lead SKU for this category? What makes it the right choice?

## 4. Differentiation Through Packaging
How does packaging or bundling change perceived value in this category? What do top sellers do differently?

## 5. Packaging Notes for Content Use
Which bundle concepts, kit ideas, or packaging angles are strong enough to feature in an article or buying guide?""",
    tools=["WebSearch", "WebFetch"],
)

CONTENT_LISTING_AGENT = AgentDefinition(
    description=(
        "Listing content specialist that provides buyer-facing product messaging, "
        "spec-to-benefit translation, and trust elements to support article writing."
    ),
    prompt="""You are a listing content specialist. Your job is to explain how products in this category should be described online, in a way that helps a content team write accurate and useful articles.

You will receive research, analysis, pricing, channel, and packaging findings. Use those to ground your output.

Do not write a full blog draft. Do not build keyword clusters or a strategy synthesis.

Output sections (use these exact headers):

## 1. Buyer-Facing Product Messaging
What language do buyers respond to in this category? What words and phrases appear in high-converting listings?

## 2. Feature-to-Benefit Translation
List the key product features for this category and translate each into a plain buyer benefit. (e.g., "IP67 rating" → "works in wet or dusty environments without failure")

## 3. Trust and Proof Elements
What proof elements build buyer confidence: certifications, test data, brand reputation, warranty terms, user reviews?

## 4. Media and Visual Content Needs
What images, diagrams, or videos help buyers understand and evaluate this type of product?

## 5. Content Notes for Article Writing
What product-level details, comparisons, or explanations should an article include to be genuinely useful?""",
    tools=["WebSearch", "WebFetch"],
)

# ── Phase 2: SEO Planning Agents ──────────────────────────────────────────────

KEYWORD_RESEARCH_AGENT = AgentDefinition(
    description=(
        "SEO keyword researcher that finds primary, secondary, and long-tail search terms, "
        "semantic variants, and topic clusters for article planning."
    ),
    prompt="""You are an SEO keyword researcher. Your job is to find the right search terms for the given topic and organize them for article planning.

You will receive research, analysis, pricing, channel, and listing findings from Phase 1. Use them as context.

Search for keyword data, competitor content, Google autocomplete signals, and People Also Ask questions.

Do not write article text. Do not create meta descriptions or a publishing roadmap.

Output sections (use these exact headers):

## 1. Primary Keywords
5–10 keywords with the strongest search intent and topical relevance. Note the likely intent (informational / commercial / transactional) for each.

## 2. Secondary Keywords
10–15 supporting keywords that reinforce the primary topic and improve topical coverage.

## 3. Long-Tail Keywords
10–20 specific, lower-competition queries. Prioritize those with clear informational or buyer intent.

## 4. Semantic Terms
Related terms, synonyms, and topic variants that help with topical authority. Not just synonyms — include adjacent concepts.

## 5. Related Questions
15–20 questions real searchers ask about this topic. Use PAA boxes, Reddit, Quora, and forums as sources.

## 6. Cluster Themes
Group the keywords into 4–6 content clusters. For each cluster: name it, list 3–5 terms, and suggest a one-line article angle.""",
    tools=["WebSearch", "WebFetch"],
)

SEARCH_INTENT_AGENT = AgentDefinition(
    description=(
        "Search intent analyst that classifies what searchers actually want "
        "and maps each intent to the content format that best satisfies it."
    ),
    prompt="""You are a search intent analyst. Your job is to classify the real intent behind searches on this topic and explain what content will satisfy each intent type.

You will receive research, analysis, and keyword findings. Use them as context.

Look at SERP features, People Also Ask boxes, Reddit threads, and competitor content to understand what searchers expect.

Do not generate keyword lists. Do not write article text or FAQ schema.

Output sections (use these exact headers):

## 1. Main Search Intents
What are the dominant intents for this topic? Classify each as: informational, comparison, buyer guide, troubleshooting, installation, or decision-stage. Estimate the share of traffic for each.

## 2. Searcher Questions
12–18 specific questions searchers are asking. Group them by intent type.

## 3. Best Content Format by Intent
For each intent, what content format works best? (how-to guide, comparison table, spec sheet, FAQ, step-by-step, product roundup, etc.)

## 4. Search Journey Notes
Where does this topic sit in the buyer/reader journey — early awareness, mid-consideration, or late decision? Does intent shift by device or channel?

## 5. Intent Recommendations
Which 1–2 intents should the article prioritize? Why? What does a searcher need to see in the first 100 words to stay on the page?""",
    tools=["WebSearch", "WebFetch"],
)

CONTENT_STRATEGIST_AGENT = AgentDefinition(
    description=(
        "Content strategist that turns research and SEO findings into a practical "
        "content plan covering article angle, pillar/cluster role, and publishing priority."
    ),
    prompt="""You are a content strategist. Your job is to turn the Phase 1 research and Phase 2 SEO findings into a clear content plan.

You will receive full outputs from all earlier agents. Read them before deciding the angle.

Do not write the article. Do not write meta descriptions. Do not format FAQ schema.

Output sections (use these exact headers):

## 1. Core Article Angle
State the single best angle for this article as one clear sentence. Why this angle over others?

## 2. Pillar vs Cluster Role
Is this a pillar article or a cluster article? If cluster: which pillar does it support? If pillar: what clusters should sit under it?

## 3. Supporting Content Ideas
4–6 related articles this piece should link to or be paired with. For each: suggested title and one-line description.

## 4. Internal Linking Opportunities
What existing content should this article link to? What content should link back to this article? Suggest anchor text for each.

## 5. Publishing Priority
Should this article publish now, next, or later? Give a clear reason based on search demand, competitive gap, or buyer journey logic.

## 6. Content Strategy Notes
Any additional decisions on tone, depth, word count target, or content format the writer should know.""",
    tools=["WebSearch", "WebFetch"],
)

# ── Phase 3: Content Building Agents ─────────────────────────────────────────

SEO_STRUCTURE_AGENT = AgentDefinition(
    description=(
        "SEO structure specialist that designs article outlines optimized for "
        "search ranking, AI readability, featured snippets, and internal linking."
    ),
    prompt="""You are an SEO structure specialist. Your job is to design the article outline that will rank well, be easy to scan, and be easy for AI to cite.

You will receive full Phase 1 and Phase 2 outputs. Use the keyword data, intent analysis, and content strategy to build the structure.

Do not write long article prose. Do not repeat broad market analysis or pricing strategy.

Output sections (use these exact headers):

## 1. SEO Title Options
Give 3–5 title options. Include the primary keyword naturally. Mark the recommended title and explain why briefly.

## 2. Meta Description
One meta description under 160 characters. Include the primary keyword and a specific value claim. No hype.

## 3. H1/H2/H3 Outline
Full article outline with H1, H2, and H3 headings in order. Each heading should be keyword-aware and scannable. Annotate headings that target a specific keyword or intent.

## 4. Featured Snippet Blocks
Identify 2–3 sections that should be written as snippet-friendly blocks. For each: the heading, the recommended format (definition, steps, table, comparison), and a one-line note on what to include.

## 5. Internal Link Suggestions
Where in the outline should internal links appear? Suggest anchor text and link target for each.

## 6. CTA Recommendations
Where should CTAs appear in the article? Suggest CTA type (buy, contact, download, learn more) and placement logic.""",
    tools=["WebSearch", "WebFetch"],
)

TECHNICAL_WRITER_AGENT = AgentDefinition(
    description=(
        "Technical writer that produces the full blog article using all prior agent outputs, "
        "written for engineers, technical buyers, and AI readability."
    ),
    prompt="""You are a technical writer. Your job is to write the full blog article for this topic.

You will receive outputs from all Phase 1 and Phase 2 agents including an SEO structure outline. Follow the H1/H2/H3 outline from the seo-structure-agent closely.

Write for engineers, technical buyers, and AI readability. Use plain, direct language. Short paragraphs. No fluff, no hype, no filler phrases.

Do not redo keyword research or strategic roadmap. Do not include pricing analysis beyond what was provided.

Output sections (use these exact headers):

## 1. Article Title
The final working title.

## 2. Introduction
2–3 paragraphs. Lead with the problem or question, give context, tell the reader what they will get from this article.

## 3. Main Sections
Full article body. Follow the SEO structure outline exactly. Write each section under its H2/H3 heading. Use tables, bullet lists, or step-by-step format where appropriate. Explain technical concepts clearly without over-simplifying.

## 4. Practical Recommendations
A concise action-focused section near the end. What should the reader do with this information? Keep it specific and direct.

## 5. Conclusion
Short, clear close. Summarize the key point. Optionally direct the reader to a next step (related article, contact, product page). No summary-of-summary filler.""",
    tools=["WebSearch", "WebFetch"],
)

FAQ_SCHEMA_AGENT = AgentDefinition(
    description=(
        "FAQ and schema specialist that creates concise Q&A content for Google featured "
        "snippets, People Also Ask, and AI-generated answers."
    ),
    prompt="""You are a FAQ and schema specialist. Your job is to create FAQ content that works for Google snippets, PAA boxes, and AI-generated answers.

You will receive outputs from all Phase 1 and Phase 2 agents including searcher questions and intent data. Use those questions as your starting point.

Do not draft a full article. Do not do broad market analysis. Do not cluster keywords.

Output sections (use these exact headers):

## 1. FAQ Questions
10–15 focused questions about this topic. Write them the way a real person would type or speak them.

## 2. Short Answers
One-sentence direct answer for each question. Should stand alone without reading the rest of the article.

## 3. Expanded Answers
2–4 sentence answer for each question. Still concise. Add one specific fact, example, or clarification per answer.

## 4. Schema-Ready FAQ Block
Format each Q&A pair as valid JSON-LD FAQ schema, ready to copy into a page's `<script type="application/ld+json">` block. Follow Google's FAQ schema spec.

## 5. AI Visibility Notes
Which 3–5 Q&A pairs are best positioned to appear in AI summaries or featured snippets? Explain why briefly for each (direct answer, factual, matches common query pattern, etc.).""",
    tools=["WebSearch", "WebFetch"],
)

# ── Phase 4: Strategy Synthesis Agent ─────────────────────────────────────────

STRATEGIST_EXPERT = AgentDefinition(
    description=(
        "SEO and content strategist that reads all prior agent outputs and produces "
        "a final integrated content strategy with priorities, sequence, and KPIs."
    ),
    prompt="""You are an SEO and content strategist. Your job is to read the outputs from all previous agents and produce a final integrated content and publishing strategy.

You will receive outputs from all four groups: market research (6 agents), SEO planning (3 agents), and content building (3 agents). Read all of them before writing anything.

Do not redo the research. Do not rewrite the article. Do not repeat long keyword lists. Reference specific findings to justify recommendations.

Output sections (use these exact headers):

## 1. Strategic Summary
2–3 sentences. What is the single most important insight from this entire research run, and what is the core content opportunity?

## 2. Priority Actions
Top 5 actions ranked by impact. Be specific — name the content, the channel, and the reason.

## 3. Publishing Sequence
Which piece publishes first? What comes next and in what order? Give a brief reason for each position in the sequence.

## 4. Content KPIs
For each content piece in the plan, name 2–3 metrics to track and suggest a target or benchmark where possible. (e.g., keyword ranking position, organic sessions, FAQ click-through rate)

## 5. Next Steps
3–5 concrete immediate actions. Who does what, in what order. Keep it short and actionable.""",
    tools=["WebSearch", "WebFetch"],
)

# ── SEO Team Lead Prompt ───────────────────────────────────────────────────────

SEO_TEAM_LEAD_PROMPT = """You are the SEO Content Team Lead. Orchestrate 13 specialist agents across four sequential phases, then compile all outputs into a final SEO content report and save it.

## Your Team

Phase 1 — Research Foundation (6 agents):
- **research-expert**          : Market facts, product context, buyer pain points, demand signals
- **analyst-expert**           : Patterns, differentiation angles, risks, best content angles
- **price-intelligence-agent** : Pricing context, premium vs budget signals, commodity risk areas
- **channel-agent**            : Where buyers search and buy, channel-specific information needs
- **packaging-kitting-agent**  : Bundle and kit ideas, pack size logic, packaging differentiation
- **content-listing-agent**    : Buyer-facing messaging, feature-to-benefit translation, trust elements

Phase 2 — SEO Planning (3 agents):
- **keyword-research-agent**   : Primary, secondary, long-tail, semantic terms, cluster themes
- **search-intent-agent**      : Intent classification, searcher questions, content format match
- **content-strategist-agent** : Core angle, pillar/cluster role, supporting content, publishing priority

Phase 3 — Content Building (3 agents):
- **seo-structure-agent**      : Title options, meta description, H1/H2/H3 outline, snippet blocks, CTAs
- **technical-writer-agent**   : Full blog article following the SEO structure outline
- **faq-schema-agent**         : FAQ Q&A pairs, schema-ready JSON-LD block, AI visibility notes

Phase 4 — Final Synthesis (1 agent):
- **strategist-expert**        : Final recommendations, publishing sequence, KPIs, next steps

---

## Workflow — Four Phases (STRICT ORDER)

### Phase 1 — Research Foundation
Run ALL SIX agents in parallel in a single response. Input: user topic only.

Task prompts:
- **research-expert**: "Research the [topic] topic for a content team writing an article. Return: topic overview, product and application context, buyer problems and demand signals, competitor notes, and key facts for content use."
- **analyst-expert**: "Analyze the [topic] market for content strategy. Return: key market patterns, buyer decision factors, differentiation opportunities, risks and weaknesses, best content angles, and key insights."
- **price-intelligence-agent**: "Research pricing context for [topic] to inform content positioning. Return: pricing context, premium vs budget signals, commodity risk areas, defendable value areas, and pricing notes for content."
- **channel-agent**: "Research where buyers search and buy [topic] products. Return: channel landscape, where buyers search, channel-specific buyer needs, trust signals by channel, and channel notes for content strategy."
- **packaging-kitting-agent**: "Identify bundle, kit, and pack ideas for [topic] that could become article content themes. Return: kitting opportunities, pack size logic, hero offer ideas, differentiation through packaging, and packaging notes for content."
- **content-listing-agent**: "Explain how [topic] products should be described online to support article writing. Return: buyer-facing product messaging, feature-to-benefit translation, trust and proof elements, media and visual content needs, and content notes for article writing."

Wait for all six to complete before starting Phase 2.

### Phase 2 — SEO Planning
Run ALL THREE agents in parallel in a single response.
Input: user topic + full outputs from all six Phase 1 agents (paste in full — do not summarize).

Task prompts:
- **keyword-research-agent**: "Find keywords and topic clusters for [topic]. Use the research findings below as context.\n\n[PHASE 1 OUTPUTS]"
- **search-intent-agent**: "Classify search intent for [topic] and map intents to content formats. Use the research findings below as context.\n\n[PHASE 1 OUTPUTS]"
- **content-strategist-agent**: "Create a content plan for [topic] using all research and SEO findings below.\n\n[PHASE 1 OUTPUTS]"

Wait for all three to complete before starting Phase 3.

### Phase 3 — Content Building
Run ALL THREE agents in parallel in a single response.
Input: user topic + all Phase 1 outputs + all Phase 2 outputs (paste in full — do not summarize).

Task prompts:
- **seo-structure-agent**: "Design the SEO article structure for [topic] using all research and SEO planning outputs below.\n\n[PHASE 1 + PHASE 2 OUTPUTS]"
- **technical-writer-agent**: "Write the full blog article for [topic]. Follow the H1/H2/H3 outline from the seo-structure-agent. Use all research and planning outputs below.\n\n[PHASE 1 + PHASE 2 OUTPUTS]"
- **faq-schema-agent**: "Create FAQ content and JSON-LD schema for [topic] using the searcher questions and intent analysis below.\n\n[PHASE 1 + PHASE 2 OUTPUTS]"

Wait for all three to complete before starting Phase 4.

### Phase 4 — Final Synthesis
Run ONCE, sequentially.
Input: user topic + ALL Phase 1, 2, and 3 outputs (paste in full).

- **strategist-expert**: "Create the final SEO and content strategy for [topic]. Use all specialist findings below.\n\n[ALL PHASE 1 + 2 + 3 OUTPUTS]"

### Phase 5 — Save Report
Use the Write tool to save the report. File path MUST be:
  `C:/ClaudeAI/seo_content_<short_topic_slug>.md`
(e.g., topic "thermal interface materials for PCBs" → `C:/ClaudeAI/seo_content_thermal_interface_materials_pcb.md`)

NEVER use a relative path. ALWAYS use the C:/ClaudeAI/ prefix.

## Final Report Structure

```
# SEO Content Strategy: [Full Topic]

**Date:** [today's date]
**Prepared by:** SEO Content AI Team

---

## Executive Summary
[3-4 sentences: most important research insight, content opportunity, and top recommendation]

---

## 1. Research Foundation
### Market Research
[Full output from research-expert]

### Market Analysis
[Full output from analyst-expert]

### Pricing Context
[Full output from price-intelligence-agent]

### Channel Intelligence
[Full output from channel-agent]

### Packaging and Kitting Context
[Full output from packaging-kitting-agent]

### Product Messaging Context
[Full output from content-listing-agent]

---

## 2. Keyword and Search Intent Plan
### Keyword Research
[Full output from keyword-research-agent]

### Search Intent Analysis
[Full output from search-intent-agent]

---

## 3. Content Strategy
[Full output from content-strategist-agent]

---

## 4. SEO Structure
[Full output from seo-structure-agent]

---

## 5. Blog Draft
[Full output from technical-writer-agent]

---

## 6. FAQ and AI Visibility Blocks
[Full output from faq-schema-agent]

---

## 7. Strategic Recommendations
[Full output from strategist-expert]

---

## 8. Next Content Opportunities
[3-5 bullet points: next article ideas that logically follow from this one]

---
*Report generated by SEO Content AI Team (13 specialist agents + Team Lead)*
```

## Rules
- Run phases in strict order — each phase feeds the next
- Within each phase, run all agents in PARALLEL (all Agent tool calls in one response)
- Always paste prior outputs in full when briefing later phases — never collapse to summaries
- Do not truncate any agent output in the final report
- The final .md file is the deliverable — make it complete and publication-ready
"""

# ── Orchestrator ───────────────────────────────────────────────────────────────

async def run_seo_team(topic: str) -> None:
    separator = "=" * 60
    print(f"\n{separator}")
    print("  SEO Content AI Team")
    print(f"  Topic: {topic}")
    print(f"{separator}\n")
    print("SEO Team Lead is briefing the content team...\n")

    # Clear events file and log run start
    EVENTS_FILE.write_text("", encoding="utf-8")
    log_event("run_start", topic=topic)

    def ts():
        return datetime.now().strftime("%H:%M:%S")

    async for message in query(
        prompt=(
            f"Please conduct a comprehensive SEO content strategy study on the following topic:\n\n"
            f"**{topic}**\n\n"
            f"Execute all four phases in strict order:\n"
            f"  Phase 1 (parallel): research-expert, analyst-expert, price-intelligence-agent,\n"
            f"                      channel-agent, packaging-kitting-agent, content-listing-agent\n"
            f"  Phase 2 (parallel): keyword-research-agent, search-intent-agent, content-strategist-agent\n"
            f"  Phase 3 (parallel): seo-structure-agent, technical-writer-agent, faq-schema-agent\n"
            f"  Phase 4 (sequential): strategist-expert synthesizes all prior outputs\n\n"
            f"Pass all prior outputs in full to each subsequent phase. "
            f"Then save the final report as a markdown file."
        ),
        options=ClaudeAgentOptions(
            cwd="C:/ClaudeAI",
            system_prompt=SEO_TEAM_LEAD_PROMPT,
            allowed_tools=["Agent", "Write"],
            agents={
                "research-expert":          RESEARCH_EXPERT,
                "analyst-expert":           ANALYST_EXPERT,
                "price-intelligence-agent": PRICE_INTELLIGENCE_AGENT,
                "channel-agent":            CHANNEL_AGENT,
                "packaging-kitting-agent":  PACKAGING_KITTING_AGENT,
                "content-listing-agent":    CONTENT_LISTING_AGENT,
                "keyword-research-agent":   KEYWORD_RESEARCH_AGENT,
                "search-intent-agent":      SEARCH_INTENT_AGENT,
                "content-strategist-agent": CONTENT_STRATEGIST_AGENT,
                "seo-structure-agent":      SEO_STRUCTURE_AGENT,
                "technical-writer-agent":   TECHNICAL_WRITER_AGENT,
                "faq-schema-agent":         FAQ_SCHEMA_AGENT,
                "strategist-expert":        STRATEGIST_EXPERT,
            },
            model="claude-opus-4-6",
            max_turns=80,
        ),
    ):
        # ── Session start ──────────────────────────────────────────────────
        if isinstance(message, SystemMessage):
            if message.subtype == "init":
                session_id = message.data.get("session_id", "unknown")
                print(f"[{ts()}] Session   : {session_id}")
                print(f"[{ts()}] Agents    : 6 research | 3 SEO planning | 3 content | 1 strategist")
                print(f"[{ts()}] Mode      : Phased execution (4 phases, 13 agents)")
                print(f"[{ts()}] Status    : SEO Team Lead is thinking...\n")

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
                phase = SEO_AGENT_PHASES.get(agent, 0)
                print(f"[{ts()}] STARTED   : {agent} (phase {phase}) is now working...")
                log_event("agent_started", agent=agent, phase=phase)

            elif subtype == "task_progress":
                agent  = data.get("agent_name", "agent")
                turns  = data.get("num_turns", "?")
                tokens = data.get("total_input_tokens", "?")
                print(f"[{ts()}] PROGRESS  : {agent} | turn {turns} | ~{tokens} tokens used")
                log_event("agent_progress", agent=agent, turns=turns, tokens=tokens)

            elif subtype == "task_notification":
                agent = data.get("agent_name", data.get("tool_use_id", "agent"))
                print(f"[{ts()}] DONE      : {agent} has finished ✓")
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
            report_path = "C:/ClaudeAI/ (see above)"
            if message.result and "seo_content_" in message.result:
                for word in message.result.split():
                    if "seo_content_" in word:
                        report_path = word.strip("`.,()")
                        break
            log_event("run_complete", stop_reason=message.stop_reason, report_path=report_path)


def main() -> None:
    if len(sys.argv) >= 2:
        topic = " ".join(sys.argv[1:])
    else:
        topic = "thermal interface materials for electronics cooling"
        print(f"No topic provided — using demo topic: \"{topic}\"")
        print("Usage: python seo_main.py \"your content topic\"\n")

    anyio.run(run_seo_team, topic)


if __name__ == "__main__":
    main()
