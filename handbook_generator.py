"""
handbook_generator.py
─────────────────────
Pipeline for RAG-powered handbook generation using Ollama + ChromaDB.

Yield protocol from `generate_handbook_stream`:
  {"type": "plan",         "total": int, "writing_plan": str}
  {"type": "token",        "token": str, "section_index": int, "total": int}
  {"type": "section_done", "section_index": int, "total": int,
           "section_text": str, "writing_plan": str}
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Generator, Iterator
from langchain_ollama import ChatOllama
from rag_pipeline import get_db, CHROMA_HANDBOOK

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
EMBED_MODEL      = "nomic-embed-text"
CHAT_MODEL       = "phi3"
DEFAULT_CHUNK_K  = 20
SECTION_WORD_MIN = 50
SECTION_WORD_MAX = 100
PLAN_LINES_MIN   = 3
PLAN_LINES_MAX   = 10

# Matches: "1. Introduction - Main Point: Overview of the topic"
# FIX: old regex searched for "Section \d+" which the prompt explicitly forbids,
#      so parse_plan_lines always returned [] and no sections were ever generated.
_PLAN_LINE_RE = re.compile(
    r"^\s*(\d+)[\.\)]\s+(?P<title>.+?)(?:\s+-\s+Main\s+Point:|\s*:\s*)(?P<point>.+)$",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Singletons — model + embeddings only
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_model() -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL, streaming=True)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_knowledge(topic, k=DEFAULT_CHUNK_K):
    """Return the top-k most relevant document chunks for *topic*."""
    db = get_db(CHROMA_HANDBOOK)
    results = db.similarity_search(topic, k=k)
    return [doc.page_content for doc in results]


# ─────────────────────────────────────────────────────────────────────────────
# Step I — Plan  (blocking; full plan needed before any section is written)
# ─────────────────────────────────────────────────────────────────────────────

def plan_handbook(topic: str, knowledge_chunks: list[str]) -> str:
    """
    Ask the LLM to build a structured outline and return it as a raw string.

    Every section line must match _PLAN_LINE_RE, e.g.:
        1. Introduction - Main Point: Sets the context for the handbook
    """
    model   = get_model()
    context = "\n\n".join(knowledge_chunks)

    # FIX: original code had a leading space before `prompt =`, causing
    #      an IndentationError at import time.
    prompt = f"""
You are an expert technical writer. Based on the knowledge below, create a
professional handbook outline for: "{topic}".

Knowledge base:
{context}

Return exactly {PLAN_LINES_MIN} to {PLAN_LINES_MAX} lines — one line per section.
STRICT FORMAT (VERY IMPORTANT):
Each line MUST follow EXACTLY this pattern:
1. Title - Main Point: Description

Examples:
1. Introduction - Main Point: Overview of the topic
2. Safety Rules - Main Point: Key workplace guidelines

If format is incorrect, the system will FAIL.

Rules:
- N is the section number (1, 2, 3, …).
- Do NOT write the words "Section" or "Chapter" in the title.
- Keep titles concise and academic.
- Do not include word counts.
- Do not include references in this outline.
- Make the outline suitable for a handbook with a table of contents and a reference page.
- Output ONLY the numbered lines — no preamble, no commentary, no blank lines between entries.
"""
    return model.invoke(prompt).content


# ─────────────────────────────────────────────────────────────────────────────
# Step II — Write one section (streaming)
# ─────────────────────────────────────────────────────────────────────────────

def stream_section(
    topic: str,
    writing_plan: str,
    section_title: str,
    section_point: str,
    last_section: str = "",
) -> Iterator[str]:
    """
    Stream one handbook section token-by-token.

    Args:
        section_title:  Clean title extracted from the plan line (no number prefix).
        section_point:  The "Main Point" description from the plan line.
    """
    model = get_model()

    prior_context = (
        f"Previous section (for continuity — do NOT repeat it):\n{last_section}\n\n"
        if last_section.strip()
        else "This is the first section — no prior text yet.\n\n"
    )

    prompt = f"""
You are an expert technical writing assistant creating a professional handbook
on: "{topic}"

Full writing plan:
{writing_plan}

{prior_context}
Now write ONLY the following section:
Title : {section_title}
Goal  : {section_point}

Requirements:
- Begin with this exact heading (no number prefix): ## {section_title}
- Write only this section — do not reproduce any earlier section.
- Target {SECTION_WORD_MIN}–{SECTION_WORD_MAX} words.
- Use clear prose, concrete examples, and bullet points where appropriate.
- Do not mention word count inside the section.
"""
    # FIX: original streamed `chunk.content` which is correct, but passed the
    #      full `section_plan_line` (including "1. Title - Main Point: ...") as
    #      the heading, producing "## 1. Title - Main Point: ..." in the output.
    #      Now we pass pre-parsed title and point separately.
    for chunk in model.stream(prompt):
        token = chunk.content
        if token:
            yield token


# ─────────────────────────────────────────────────────────────────────────────
# Step III — Compile final document
# ─────────────────────────────────────────────────────────────────────────────

def compile_handbook(
    topic: str,
    sections: list[str],
    plan_lines: list[dict],
) -> str:
    """
    Assemble title, table of contents, and body into the final markdown document.

    Args:
        plan_lines: Pre-parsed list of dicts with keys "title" and "point",
                    as returned by parse_plan_lines().  Passing them in avoids
                    parsing the plan string a second time.
    """
    title_block = f"# Handbook on {topic}\n\n"

    # FIX: old code searched for 'Title:\s*...' inside the raw plan line string,
    #      but the format is "N. Title - Main Point: ..." so the regex never
    #      matched and every TOC entry fell back to "Section N".
    toc_lines = ["## Table of Contents\n"]
    for i, entry in enumerate(plan_lines, start=1):
        toc_lines.append(f"{i}. {entry['title']}")

    toc  = "\n".join(toc_lines)
    body = "\n\n---\n\n".join(sections)

    return f"{title_block}{toc}\n\n---\n\n{body}"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_plan_lines(writing_plan: str) -> list[dict]:
    """
    Parse the LLM plan into a list of structured dicts.

    Each dict contains:
        "number" (int)  — section number from the plan
        "title"  (str)  — section title
        "point"  (str)  — main-point description

    Lines that do not match the expected format are silently skipped,
    which makes the parser robust to minor LLM formatting deviations.

    FIX: the original function searched for "Section \\d+" which the prompt
         explicitly forbids, so it always returned [], causing the generator
         to yield zero sections.
    """
    print("RAW PLAN:\n", writing_plan)
    entries: list[dict] = []
    for line in writing_plan.splitlines():
        m = _PLAN_LINE_RE.match(line)
        if m:
            entries.append({
                "number": int(m.group(1)),
                "title":  m.group("title").strip(),
                "point":  m.group("point").strip(),
            })
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def generate_handbook_stream(topic: str) -> Generator[dict, None, None]:
    """
    Full RAG → Plan → Stream pipeline.

    Yields dicts according to the protocol described in the module docstring.
    """  
    # Step 1: retrieve knowledge safely
    try:
      chunks = retrieve_knowledge(topic)
      print(f"Retrieved {len(chunks)} chunks")
    except Exception as e:
      print(f"⚠️ Retrieval failed: {e}")
      chunks = []

    # Step 2: generate plan
    writing_plan = plan_handbook(topic, chunks)
    print("\n=== RAW PLAN ===")
    print(writing_plan)
    print("================\n")

    # Step 3: parse
    plan_lines = parse_plan_lines(writing_plan)
    print("\n=== PARSED PLAN ===")
    print(plan_lines)
    print("===================\n")
    
    total = len(plan_lines)

    if total == 0:
      print("⚠️ Parsing failed — using fallback plan")

      # fallback: build plan from raw lines
      raw_lines = [
        l.strip() for l in writing_plan.splitlines()
        if l.strip()
      ]

      plan_lines = []
      for i, line in enumerate(raw_lines):
          plan_lines.append({
            "number": i + 1,
            "title": line,
            "point": line,
          })

      total = len(plan_lines)

    yield {"type": "plan", "total": total, "writing_plan": writing_plan , "plan_lines": plan_lines}

    last_section = ""
    sections: list[str] = []

    for i, entry in enumerate(plan_lines, start=1):
        section_tokens: list[str] = []

        for token in stream_section(
            topic=topic,
            writing_plan=writing_plan,
            section_title=entry["title"],
            section_point=entry["point"],
            last_section=last_section,
        ):
            section_tokens.append(token)
            yield {"type": "token", "token": token, "section_index": i, "total": total}

        section_text = "".join(section_tokens)
        sections.append(section_text)
        last_section = section_text          # keep only the previous section in memory

        yield {
            "type":          "section_done",
            "section_index": i,
            "total":         total,
            "section_text":  section_text,
            "writing_plan":  writing_plan,
        }
