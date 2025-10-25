import PyPDF2

class PDFLoader:
    """Load PDF and split into paragraphs (documents)."""

    @staticmethod
    def load_pdf(file_path: str) -> list:
        print(f"[PDFLoader] Loading PDF: {file_path}")
        docs = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                docs.extend(paragraphs)
                print(f"[PDFLoader] Page {page_num+1}: {len(paragraphs)} paragraphs added")
        print(f"[PDFLoader] Total documents extracted: {len(docs)}")
        return docs
