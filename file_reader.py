from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def read_pdf_text(path: str | Path) -> str:
    """read and return the text content of a pdf file"""
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = PyPDFLoader(pdf_path).load()
    texts = [doc.page_content for doc in docs]
    return "\n".join(texts)


def read_multiple_pdf(path: list[str | Path]) -> list[Document]:
    """read text from multiple pdf and return a list of documents"""
    docs: list[Document] = []

    for pdf in path:
        pdf_path = Path(pdf)
        texts = read_pdf_text(Path(pdf_path))
        docs.append(Document(page_content=texts, metadata={"source": str(pdf_path)}))

    return docs


docs_path: list[str | Path] = ["./pdf/climate_change.pdf", "./pdf/global_warming.pdf"]
