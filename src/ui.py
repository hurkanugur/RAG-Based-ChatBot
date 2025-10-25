import gradio as gr
from src.pdf_loader import PDFLoader
from src.generator import Generator
import torch

class ChatbotUI:
    def __init__(self, device: torch.device):
        self.documents = []
        self.generator = None
        self.device = device

    # Load PDF
    def load_pdf(self, file):
        if file is None or not file.name.lower().endswith(".pdf"):
            self.documents = []
            return gr.update(value="⚠️ Please upload a valid PDF.", interactive=False), gr.update(interactive=False)

        self.documents = PDFLoader.load_pdf(file.name)
        return gr.update(value=f"✅ PDF loaded with {len(self.documents)} documents.", interactive=False), gr.update(interactive=True)

    # Ask question (lazy load model)
    def ask_question(self, question):
        if not self.documents:
            return "⚠️ PDF not loaded yet."
        if not question.strip():
            return "⚠️ Please type a question."

        # Load model on first ask
        if self.generator is None:
            print(f"[ChatbotUI] Loading model on device: {self.device} ...")
            self.generator = Generator(self.documents, device=self.device)
            print("[ChatbotUI] Model loaded!")

        return self.generator.generate_answer(question)

    # Launch UI
    def launch(self):
        with gr.Blocks(theme=gr.themes.Ocean(), title="📄 PDF RAG ChatBot") as app:
            gr.Markdown(
                """
                # 📄 RAG-Based ChatBot (Offline) 
                Upload a PDF and ask questions! The chatbot will answer based on the PDF content.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_file = gr.File(label="📁 Upload PDF", file_types=[".pdf"])
                    user_input = gr.Textbox(label="📝 Ask a question", placeholder="Type your question here...")
                    ask_btn = gr.Button("🤖 Ask", interactive=False, variant="primary")
                    clear_btn = gr.Button("🧹 Clear", variant="secondary")

                with gr.Column(scale=1):
                    status = gr.Textbox(
                        label="Status / Answer",
                        placeholder="Answers will appear here...",
                        interactive=False,
                        lines=11,
                        show_copy_button=True,
                    )

            # Connect file upload
            def enable_ask_btn(file_info):
                _, enable = self.load_pdf(file_info)
                return enable

            pdf_file.change(
                self.load_pdf,
                inputs=pdf_file,
                outputs=[status, ask_btn]
            )

            # Connect Ask button
            ask_btn.click(
                self.ask_question,
                inputs=user_input,
                outputs=status
            )

            # Clear inputs
            clear_btn.click(
                lambda: ("", ""),
                inputs=None,
                outputs=[user_input, status]
            )

            gr.Markdown(
                """
                ---
                💡 **Tip:** Upload PDFs with clear paragraphs for better answers.  
                ⚠️ **Note:** The first question may take some time because the model is loaded on-demand. 

                ---
                👨‍💻 **Developed by [Hürkan Uğur](https://github.com/hurkanugur)**  
                🔗 Source Code: [RAG-Based ChatBot (Offline)](https://github.com/hurkanugur/RAG-Based-ChatBot)
                """
            )

        app.launch()