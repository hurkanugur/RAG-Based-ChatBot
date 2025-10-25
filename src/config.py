# ============================
# 🧠 Model Configuration
# ============================

# Sentence embedding model used to convert text into numerical vector form
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

# Text generation model used to create natural language answers from retrieved context
GENERATION_MODEL = "google/flan-t5-base"

# ============================
# 🔍 Retriever Settings
# ============================

# TOP_K → Number of most relevant text chunks (documents) retrieved from the PDF
# Example: TOP_K = 1 means the model will pick the top 1 most similar pieces of text
TOP_K = 1

# ============================
# ✍️ Generator Settings
# ============================

# MAX_TOKENS → Maximum number of tokens (≈ words or word pieces) the model can generate for the answer
# For example, 200 tokens ≈ 150–200 words depending on language and tokenizer
MAX_TOKENS = 200

# MAX_LENGTH → Maximum number of tokens for the *input* (question + context)
# 512 tokens roughly equals about 350–400 words, not characters
MAX_LENGTH = 512
