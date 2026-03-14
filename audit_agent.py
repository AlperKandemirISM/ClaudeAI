#!/usr/bin/env python3
"""
Product Content Audit Agent — schlemmerturkiye.com
===================================================
Crawls product pages and flags missing meta descriptions,
empty product text, and part numbers absent from titles.

Usage:
  python audit_agent.py 9800808   # Audit a single product by part number
  python audit_agent.py           # Audit all products on the site
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

BASE_URL = "https://www.schlemmerturkiye.com"

# ── Specialist Agent Definitions ──────────────────────────────────────────────

CRAWLER_AGENT = AgentDefinition(
    description=(
        "Web crawler that discovers product page URLs on schlemmerturkiye.com. "
        "Can locate a single product by part number or enumerate all products via sitemap/categories."
    ),
    prompt=f"""You are a focused web crawler for {BASE_URL}.

Your job depends on the input you receive:

A) SINGLE PRODUCT MODE — you receive a part number (e.g. 9800808):
   1. Try fetching {BASE_URL}/sitemap.xml to check if it exists and lists product URLs.
   2. Search the sitemap or try common URL patterns to find the product page:
      - {BASE_URL}/urun/9800808
      - {BASE_URL}/product/9800808
      - {BASE_URL}/tr/urun/9800808
   3. If sitemap doesn't help, use WebSearch: site:schlemmerturkiye.com 9800808
   4. Return the exact product URL.

B) ALL PRODUCTS MODE — you receive no part number:
   1. Fetch {BASE_URL}/sitemap.xml — if a sitemap index, follow sub-sitemaps.
   2. Extract all product page URLs (typically /urun/ or /product/ paths).
   3. If sitemap is missing, fetch the homepage, find category links, follow them,
      and collect individual product page URLs from listings.
   4. Cap at 50 products to keep the audit manageable.

Output format — always end with this block (one URL per line):
PRODUCT_URLS_START
https://www.schlemmerturkiye.com/...  | PART: 9800808
https://www.schlemmerturkiye.com/...  | PART: unknown
PRODUCT_URLS_END

Only include real product pages, not category or static pages.""",
    tools=["WebFetch", "WebSearch"],
)

AUDITOR_AGENT = AgentDefinition(
    description=(
        "Content auditor that fetches product pages and checks each for: "
        "meta description, product description text, and part number in the title."
    ),
    prompt="""You are a precise product page content auditor.

For EACH product URL you receive:
1. Fetch the page with WebFetch.
2. Parse the HTML and check all three criteria:

   A. META DESCRIPTION
      - Look for <meta name="description" content="...">
      - PASS if present AND content is ≥ 50 characters and not generic/empty.
      - FAIL if missing, empty, or shorter than 50 characters.

   B. PRODUCT TEXT
      - Look for substantial descriptive text in the product section
        (description paragraphs, feature lists, specification text).
      - PASS if there is ≥ 100 characters of product-specific body text.
      - FAIL if the page only shows a title/heading with no description.

   C. PART NUMBER IN TITLE
      - Check the <title> tag and the main H1/H2 heading.
      - PASS if the part number (numeric code) appears in either.
      - FAIL if neither contains a recognisable product/part number.

Output one audit block per product, separated by ---:

---
URL: https://www.schlemmerturkiye.com/...
PART_NUMBER: 9800808
PAGE_TITLE: [full <title> content]
META_DESCRIPTION: PASS | "[first 120 chars]"
PRODUCT_TEXT: PASS | [brief note, e.g. "340 chars of description found"]
PART_NUMBER_IN_TITLE: PASS | [note]
OVERALL: PASS
ISSUES: none
---

---
URL: https://www.schlemmerturkiye.com/...
PART_NUMBER: unknown
PAGE_TITLE: [full <title> content]
META_DESCRIPTION: FAIL | missing
PRODUCT_TEXT: FAIL | only heading found, no description text
PART_NUMBER_IN_TITLE: FAIL | no numeric code in title or H1
OVERALL: ISSUES_FOUND
ISSUES: missing meta description, no product text, part number absent from title
---

Be thorough. Fetch every URL. Never skip a product.""",
    tools=["WebFetch"],
)

# ── Lead Orchestrator Prompt ──────────────────────────────────────────────────

def build_lead_prompt(product_number: str | None) -> str:
    if product_number:
        mode_desc  = f"a **single product** — part number `{product_number}`"
        crawl_task = (
            f"Find the product page for part number {product_number} on "
            f"{BASE_URL}. Return its URL."
        )
    else:
        mode_desc  = "**all products** on the site"
        crawl_task = (
            f"Crawl {BASE_URL} to discover all product page URLs "
            f"(use sitemap and/or category pages). Return every URL found (max 50)."
        )

    return f"""You are the Product Content Audit Lead for {BASE_URL}.

## Mission
Audit {mode_desc} for three content quality issues:
- Missing or inadequate meta descriptions
- Missing product description text
- Part number absent from the page title

## Your Team
- **crawler-agent** : Discovers product URLs on the site
- **auditor-agent** : Audits each product page for the three content criteria

## Workflow

### Step 1 — Discover Products
Call **crawler-agent** with this task:
"{crawl_task}"

### Step 2 — Audit Products
Extract all URLs from the PRODUCT_URLS_START / PRODUCT_URLS_END block in the
crawler's response, then call **auditor-agent** with all the URLs.

If there are more than 10 products, split them into batches of 10 and call
**auditor-agent** multiple times IN PARALLEL (multiple Agent tool calls at once).

### Step 3 — Compile & Save Report
Synthesise all auditor results into the final report structure below, then use
the Write tool to save it to:
  `C:/ClaudeAI/audit_schlemmer_<DATE>.md`
where <DATE> is today's date (e.g. 2026-03-14).

NEVER use relative paths or /home/user/. Always use C:/ClaudeAI/ prefix.

## Final Report Structure

```
# Product Content Audit — schlemmerturkiye.com

**Date:** [today]
**Mode:** [Single Product: XXXXXX | Full Site Audit]
**Products Audited:** [N]  |  **Passed:** [N]  |  **Issues Found:** [N]

---

## Executive Summary
[2–3 sentences: overall site health, most common issue type, top priority fix]

---

## Products with Issues

| # | URL | Part Number | Meta Desc | Product Text | Part# in Title | Issues |
|---|-----|-------------|-----------|--------------|----------------|--------|
| 1 | [url] | 9800808 | FAIL | PASS | FAIL | missing meta desc, part# absent from title |
...

---

## Passing Products

| # | URL | Part Number |
|---|-----|-------------|
| 1 | [url] | 9800808 |
...

---

## Issue Breakdown

| Issue | Count | % of Products |
|-------|-------|---------------|
| Missing / short meta description | N | X% |
| Missing product text | N | X% |
| Part number absent from title | N | X% |

---

## Recommendations

1. [Most impactful fix — be specific]
2. [Second fix]
3. [Third fix]

---
*Audit generated by Product Content Audit Agent on [date]*
```

## Rules
- Always call crawler-agent FIRST to get URLs.
- Never skip the auditor step — every URL must be checked.
- Save a complete, actionable .md report — it is the deliverable.
"""


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_audit(product_number: str | None) -> None:
    sep  = "=" * 60
    mode = f"Single Product: {product_number}" if product_number else "Full Site Audit"
    print(f"\n{sep}")
    print("  Product Content Audit Agent")
    print(f"  Site : {BASE_URL}")
    print(f"  Mode : {mode}")
    print(f"{sep}\n")

    def ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    if product_number:
        prompt = (
            f"Run a product content audit on {BASE_URL}.\n\n"
            f"Mode: SINGLE PRODUCT — part number: {product_number}\n\n"
            f"1. Call crawler-agent to find the URL for part number {product_number}.\n"
            f"2. Call auditor-agent to check that page for meta description, "
            f"product text, and part number in title.\n"
            f"3. Save the audit report."
        )
    else:
        prompt = (
            f"Run a full-site product content audit on {BASE_URL}.\n\n"
            f"Mode: FULL SITE — audit all products.\n\n"
            f"1. Call crawler-agent to discover all product URLs.\n"
            f"2. Call auditor-agent (in parallel batches of 10 if needed) to check "
            f"every page for meta description, product text, and part number in title.\n"
            f"3. Save the audit report."
        )

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            cwd="C:/ClaudeAI",
            system_prompt=build_lead_prompt(product_number),
            allowed_tools=["Agent", "Write", "WebFetch", "WebSearch"],
            agents={
                "crawler-agent": CRAWLER_AGENT,
                "auditor-agent": AUDITOR_AGENT,
            },
            model="claude-opus-4-6",
            max_turns=50,
        ),
    ):
        # Session init
        if isinstance(message, SystemMessage):
            if message.subtype == "init":
                sid = message.data.get("session_id", "unknown")
                print(f"[{ts()}] Session : {sid}")
                print(f"[{ts()}] Agents  : crawler-agent | auditor-agent")
                print(f"[{ts()}] Status  : Audit Lead is initialising...\n")

        # Lead speaking
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    for line in block.text.strip().splitlines():
                        if line.strip():
                            print(f"[{ts()}] LEAD    : {line}")

        # Subagent task events
        elif hasattr(message, "subtype"):
            subtype = getattr(message, "subtype", "")
            data    = getattr(message, "data", {}) or {}

            if subtype == "task_started":
                agent = data.get("agent_name", data.get("tool_use_id", "agent"))
                print(f"[{ts()}] STARTED : {agent} is working...")

            elif subtype == "task_progress":
                agent  = data.get("agent_name", "agent")
                turns  = data.get("num_turns", "?")
                tokens = data.get("total_input_tokens", "?")
                print(f"[{ts()}] PROGRESS: {agent} | turn {turns} | ~{tokens} tokens")

            elif subtype == "task_notification":
                agent = data.get("agent_name", data.get("tool_use_id", "agent"))
                print(f"[{ts()}] DONE    : {agent} finished ✓")

        # Final result
        elif isinstance(message, ResultMessage):
            print(f"\n[{ts()}] {'='*54}")
            print(f"[{ts()}]   AUDIT COMPLETE — Report saved!")
            print(f"[{ts()}] {'='*54}")
            print(f"[{ts()}] Stop reason : {message.stop_reason}")
            if message.result:
                preview = message.result[:300].replace("\n", " ")
                print(f"[{ts()}] Preview     : {preview}...")
            print(f"\n[{ts()}] Report saved to: C:/ClaudeAI/audit_schlemmer_*.md\n")


def main() -> None:
    if len(sys.argv) >= 2:
        product_number = sys.argv[1].strip()
        print(f"Auditing single product: {product_number}")
        print(f"Usage: python audit_agent.py [product_number]\n")
    else:
        product_number = None
        print("No product number given — auditing ALL products on the site.")
        print("Usage: python audit_agent.py [product_number]\n")

    anyio.run(run_audit, product_number)


if __name__ == "__main__":
    main()
