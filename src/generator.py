from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import GENERATION_MODEL, MAX_TOKENS, MAX_LENGTH
from src.retriever import Retriever

class Generator:
    """Generate answers from queries using a local Flan-T5 model and retrieved context."""

    def __init__(self, documents=None, device=None):
        """Initialize the generator, load the model, and optionally set up a retriever."""
        self.device = device
        self.documents = documents or []
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL).to(device)
        self.retriever = None
        if self.documents:
            self.retriever = Retriever(self.documents, device=device)

    def set_documents(self, documents):
        """Update the documents and initialize a new retriever."""
        self.documents = documents
        self.retriever = Retriever(documents, device=self.device)

    def generate_answer(self, query: str) -> str:
        """Return an answer string for a query using retrieved documents as context."""
        if not self.retriever:
            return "⚠️ PDF not loaded yet."
        
        retrieved_docs = self.retriever.retrieve(query)
        context = "\n".join(retrieved_docs)
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(self.device)

        # Generate output
        outputs = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS)

        # Print raw model output tensor
        print(f"[Generator] Raw model output: {outputs}")

        # Decode to string
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[Generator] Decoded answer: {answer}")

        return answer

