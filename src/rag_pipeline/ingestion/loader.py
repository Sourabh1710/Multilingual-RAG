from pathlib import Path
from typing import Any, Dict, List

import pypdf
from pydantic import BaseModel, Field


# 1. Defining the strict Pydantic schemas for data safety
class DocumentMetadata(BaseModel):
    source_file: str
    page_number: int
    total_pages: int
    file_type: str = "pdf"
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    content: str = Field(
        ..., description="The raw text extracted from the document page"
    )
    metadata: DocumentMetadata


# 2. Implementing the Document Loader
class PDFLoader:
    def __init__(self, file_path: Path) -> None:
        """
        Initializes the PDFLoader with pathlib.Path object.
        """
        self.file_path = file_path

    async def load(self) -> List[Document]:
        """
        Asynchronously loads the pdf file and parse every page into Document object
        """
        # Ensures the file exists before trying to read it.
        if not self.file_path.exists():
            raise FileNotFoundError(f"Source file not found at {self.file_path}")

        documents: List[Document] = []

        try:
            reader = pypdf.PdfReader(self.file_path)
            total_pages = len(reader.pages)
            file_name = self.file_path.name

            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                # skip empty pages
                if not text.strip():
                    continue
                metadata = DocumentMetadata(
                    source_file=file_name,
                    page_number=page_index + 1,
                    total_pages=total_pages,
                )
                doc = Document(content=text, metadata=metadata)
                documents.append(doc)
        except Exception as e:
            raise RuntimeError(f"Failed to parse {self.file_path} : {str(e)}") from e
        return documents
