import asyncio
from pathlib import Path
from typing import List

from rag_pipeline.ingestion.chunker import Chunk, RecursiveCharacterChunker
from rag_pipeline.ingestion.loader import PDFLoader
from rag_pipeline.ingestion.preprocess import clean_and_normalize_text


class IngestionPipeline:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        max_concurrent_files: int = 5,
    ) -> None:
        """
        Initializes the ingestion pipeline with chunking and concurrency constraints.
        """
        self.chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.sem = asyncio.Semaphore(max_concurrent_files)

    async def ingest_file(self, file_path: Path) -> List[Chunk]:
        """
        Ingests a single document file: loads, normalizes text, and splits into overlapping chunks.
        """
        # Enforce rate-limits on operating system file reads using the semaphore
        async with self.sem:
            print(f"[INGEST] Loading file: {file_path.name}")

            # 1. Load the PDF document
            loader = PDFLoader(file_path)
            raw_docs = await loader.load()

            all_chunks: List[Chunk] = []

            # 2. Process and Chunk each loaded page
            for doc in raw_docs:
                # Normalize Unicode and clean whitespace
                cleaned_content = clean_and_normalize_text(doc.content)

                # Update the document's content with normalized text
                doc.content = cleaned_content

                # 3. Split the cleaned document page into chunks
                page_chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(page_chunks)

            return all_chunks

    async def ingest_directory(self, dir_path: Path) -> List[Chunk]:
        """
        Scans a directory for PDF files and processes all of them concurrently.
        """
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Ingestion directory not found at: {dir_path}")

        # Find all PDF files inside the directory (case-insensitive)
        pdf_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF"))

        if not pdf_files:
            print(f"[WARN] No PDF files found in directory: {dir_path}")
            return []

        print(
            f"[INGEST] Found {len(pdf_files)} PDF files in {dir_path.name}. Starting concurrent ingestion..."
        )

        # ingest_file tasks for each pdf file found
        tasks = [self.ingest_file(f) for f in pdf_files]

        # Run all tasks concurrently using asyncio.gather
        results = await asyncio.gather(*tasks)

        # Flatten the list of lists (List[List[Chunk]]) into a single List[Chunk]
        flat_chunks: List[Chunk] = []
        for file_chunks in results:
            flat_chunks.extend(file_chunks)

        print(
            f"[INGEST] Ingestion complete. Generated {len(flat_chunks)} chunks from {len(pdf_files)} files."
        )
        return flat_chunks
