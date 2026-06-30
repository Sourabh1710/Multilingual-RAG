import uuid
from typing import List

from pydantic import BaseModel, Field

from rag_pipeline.ingestion.loader import Document, DocumentMetadata


# 1. Define the Chunk schema
class Chunk(BaseModel):
    id: str = Field(..., description="Unique hash or string ID for this specific chunk")
    content: str = Field(..., description="The sliced text content of the chunk")
    chunk_index: int = Field(..., description="The index of this chunk in the document")
    metadata: DocumentMetadata


# 2. Implement the Chunker
class RecursiveCharacterChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        """
        Initializes the chunker with target size and overlap bounds.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Fallback separators: try split by paragraph, then single line, then words
        self.separators = ["\n\n", "\n", " ", ""]

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Splits a single Document object into a list of smaller overlapping Chunk objects.
        """
        raw_text = doc.content
        chunks: List[Chunk] = []

        # a standard splitting algorithm:
        raw_chunks = self._split_text(raw_text, self.separators)

        # Merge small raw pieces into target-sized chunks with proper overlap
        merged_texts = self._merge_splits(raw_chunks)

        for idx, text in enumerate(merged_texts):
            # Generate a clean, unique ID for this chunk (e.g., "filename_page_chunkidx")
            clean_filename = doc.metadata.source_file.replace(".", "_")
            string_id = f"{clean_filename}_p{doc.metadata.page_number}_c{idx}"

            # Convert the readable string ID into a deterministic UUID v5 string
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

            # Create the Chunk object, preserving parent metadata
            chunk = Chunk(
                id=chunk_id, content=text, chunk_index=idx, metadata=doc.metadata
            )
            chunks.append(chunk)

        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Internal recursive method that splits text using a list of fallback separators.
        """
        # Base case: if no separators left, return the text block as a single item
        if not separators:
            return [text]

        separator = separators[0]
        next_separators = separators[1:]

        # Split text by current separator
        if separator == "":
            return list(text)

        splits = text.split(separator)
        final_splits: List[str] = []

        for split in splits:
            if len(split) > self.chunk_size:
                # Recursively split the long block using remaining separators
                final_splits.extend(self._split_text(split, next_separators))
            else:
                final_splits.append(split)

        return [s for s in final_splits if s.strip()]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merges small individual splits into cohesive blocks near target chunk_size
        while respecting chunk_overlap constraints.
        """
        merged_chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # If adding this split exceeds chunk_size, save current chunk and start a new one
            if current_length + split_len > self.chunk_size:
                if current_chunk:
                    merged_chunks.append(" ".join(current_chunk))

                # Setup overlap: keep a few trailing elements from the previous chunk to maintain context flow
                overlap_chunk: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) < self.chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break

                current_chunk = overlap_chunk + [split]
                current_length = sum(len(item) for item in current_chunk)
            else:
                current_chunk.append(split)
                current_length += split_len

        # Append final chunk if anything remains
        if current_chunk:
            merged_chunks.append(" ".join(current_chunk))

        return merged_chunks
