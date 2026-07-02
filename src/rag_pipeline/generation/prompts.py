from typing import List

from rag_pipeline.ingestion.chunker import Chunk

# System prompt to set the strict grounding persona and multilingual instructions
SYSTEM_PROMPT = """
You are a helpful, strictly grounded assistant for Indian corporate employees.
Your task is to answer the user's query based ONLY on the provided Context (which is written in English).

CRITICAL INSTRUCTIONS:
1. Grounding: You must answer the user's query using ONLY the facts explicitly stated in the provided Context.
2. Honesty: If the Context does not contain the answer to the user's query, you must respond exactly with: "मुझे नहीं पता।" (I don't know). Do not attempt to guess or use your pre-trained general knowledge.
3. Multilingual Output:
   - If the user's query is in Hindi or Hinglish, answer natively in Hindi or Hinglish, but ensure technical terms (like 'casual leave', 'portal', 'calendar year') are kept intact.
   - Do not answer in English unless the query itself is strictly in English.
4. No Hallucinations: Do not add any outside details, limits, rules, or facts that are not present in the provided Context.
"""


# Helper to format the context and user query into a single string
def format_prompt(context_chunks: List[Chunk], query: str) -> str:
    """
    Formats retrieved chunks and the user's query into a standardized grounded prompt.
    """
    # Join the retrieved chunks with a clean boundary separator
    context_text = "\n\n---\n\n".join(
        [
            f"Source: {c.metadata.source_file} (Page {c.metadata.page_number})\nContent: {c.content}"
            for c in context_chunks
        ]
    )

    return f"""
{SYSTEM_PROMPT}

[Context]
{context_text}

[User Query]
{query}

Grounded Answer:
"""
