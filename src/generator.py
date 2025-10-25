from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import GENERATION_MODEL, MAX_TOKENS, MAX_LENGTH

class Generator:
    """Local RAG generator using Flan-T5."""

    def __init__(self, documents=None, device=None):
        self.device = device
        self.documents = documents or []
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL).to(device)
        self.retriever = None
        if self.documents:
            from src.retriever import Retriever
            self.retriever = Retriever(self.documents, device=device)

    def set_documents(self, documents):
        """Set documents after PDF load."""
        self.documents = documents
        from src.retriever import Retriever
        self.retriever = Retriever(documents, device=self.device)

    def generate_answer(self, query: str) -> str:
        if not self.retriever:
            return "⚠️ PDF not loaded yet."
        retrieved_docs = self.retriever.retrieve(query)
        context = "\n".join(retrieved_docs)
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
