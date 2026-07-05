import os
import asyncio
import sys
from pathlib import Path

from rag_pipeline.embeddings.pipeline import EmbeddingPipeline
from rag_pipeline.generation.pipeline import GenerationPipeline
from rag_pipeline.ingestion.pipeline import IngestionPipeline
from rag_pipeline.retrieval.pipeline import RetrievalPipeline


async def main(query: str) -> None:
    # 1. Define paths
    raw_data_dir = Path("data/raw")
    collection_name = "corporate_leave_policy"

    print()
    print("        INITIALIZING MULTILINGUAL INDIC RAG PIPELINE        ")
    print()

    # 2. Instantiate all of the asynchronous modular pipelines
    ingest_pipeline = IngestionPipeline(chunk_size=800, chunk_overlap=150)
    embed_pipeline = EmbeddingPipeline(collection_name=collection_name)
    retrieval_pipeline = RetrievalPipeline(collection_name=collection_name)
    generation_pipeline = GenerationPipeline()

    # 3. Check if there are raw PDFs to ingest first
    pdf_files = list(raw_data_dir.glob("*.pdf")) + list(raw_data_dir.glob("*.PDF"))

    if pdf_files:
        print(
            f"\n[STEP 1] Found {len(pdf_files)} PDF files in {raw_data_dir}. Processing ingestion..."
        )
        # Load, clean, and chunk the PDFs
        chunks = await ingest_pipeline.ingest_directory(raw_data_dir)

        # Embed and store the chunks into Qdrant
        await embed_pipeline.embed_and_store(chunks)
        print("[STEP 1] Ingestion and database indexing complete.")
    else:
        print(
            f"\n[STEP 1] No raw PDF files found in {raw_data_dir}. Skipping ingestion phase."
        )
        print("[STEP 1] Assuming database collection already has indexed documents...")

    # 4. Perform Retrieval based on the User Query
    print(f"\n[STEP 2] Performing semantic retrieval for query: '{query}'...")
    retrieved_chunks = await retrieval_pipeline.retrieve(query=query, top_k=2)

    if not retrieved_chunks:
        print("[STEP 2] No relevant documents found. Cannot generate answer.")
        return

    # 5. Generate Grounded Multilingual Answer
    active_provider = os.getenv("GENERATOR_PROVIDER", "gemini").upper()
    print(f"\n[STEP 3] Generating response using active provider ({active_provider})...")
    answer = await generation_pipeline.generate_answer(
        query=query, context_chunks=retrieved_chunks
    )

    print()
    print("                      FINAL GROUNDED ANSWER                    ")
    print()
    print(f"Query: {query}")
    print(f"Answer:\n{answer}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Expect the user to pass their query as a command line argument
    # Example: uv run python src/rag_pipeline/run.py "Casual leave rules kya hain?"
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a search query!")
        print('Usage: uv run python src/rag_pipeline/run.py "your query here"')
        sys.exit(1)

    user_query = sys.argv[1]
    asyncio.run(main(user_query))
