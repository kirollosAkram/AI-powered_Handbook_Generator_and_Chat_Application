import pdfplumber
from langchain_core.documents import Document

def extract_text_from_pdf(file):
    documents = []

    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file.name, "page": i}
                    )
                )

    return documents