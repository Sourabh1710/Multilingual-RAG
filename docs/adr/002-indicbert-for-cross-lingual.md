# ADR 002: Use Aligned Multilingual Embeddings for Cross-Lingual Retrieval

## Status
Accepted

## Context
The core business problem is that the corporate manuals are written entirely in English, but the users query the system natively in Hindi or code-mixed Hinglish (e.g., *"Casual leave carry forward karne ke rules kya hain?"*).

I've considered two architectural approaches to solve this:
1.  **Naive Translation Loop:** Translate the incoming Hindi/Hinglish query to English using a translation API, search the English database, generate an answer, and translate it back to Hindi.
2.  **Cross-Lingual Aligned Vector Space:** Use a single, shared multilingual embedding model that naturally maps English, Hindi, and Hinglish semantic concepts to adjacent coordinates in the same vector space.

## Decision
I chose a **Cross-Lingual Aligned Vector Space** using the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model. This completely eliminates the translation-first loop.

## Consequences

### Consequences (Pros)
- **Zero Translation Overhead:** I completely bypass the latency, API costs, and fragile dependency on external translation APIs at query time.
- **Robust Hinglish Comprehension:** Code-mixed Hinglish (which has no rigid grammar rules and destroys standard translators) is parsed natively based on semantic density, matching its English meaning accurately.
- **Instant Speed:** Querying is performed in milliseconds using direct vector geometry, rather than waiting for multiple API network hops.

### Consequences (Cons)
- **Slight Semantic Drift:** Domain-specific, highly technical jargon or uncommon regional terms can occasionally experience misalignment between languages. I mitigate this by using a high-overlap recursive chunking strategy to preserve paragraph context.
