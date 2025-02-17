import gradio as gr
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.cloud import storage

BUCKET_NAME = "ragsd-resume-bucket"
os.environ["GCS_INDEX_PATH"] = "faiss_indexes/cv_index_text-embedding-3-large"
INDEX_PATH = os.getenv("GCS_INDEX_PATH")

class CVQueryApp:
    def __init__(self):
        self.client = OpenAI()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = self._load_vector_store()
        
    def _load_vector_store(self):
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        local_path = "/tmp/faiss_index"
        blob = bucket.blob(INDEX_PATH)
        blob.download_to_filename(local_path)
        
        return FAISS.load_local(
            local_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def query(self, question: str) -> str:
        try:
            docs = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "fetch_k": 20,
                    "lambda_mult": 0.7
                }
            ).get_relevant_documents(question)

            context = "\n".join(f"[{doc.metadata['section']}]\n{doc.page_content}" for doc in docs)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a precise CV analysis assistant. Your task is to:
                        1. Only use information explicitly stated in the provided CV sections
                        2. Quote specific details when possible
                        3. If information is not found, clearly state 'Information not found in CV'
                        4. Maintain chronological accuracy when discussing experience
                        5. Consider all provided sections before answering
                        6. Use relevant links of demoes, where relevant, to emphasise skills"""},
                    {"role": "user", "content": f"Based on these CV sections:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"

def create_gradio_interface():
    cv_app = CVQueryApp()
    
    demo = gr.Interface(
        fn=cv_app.query,
        inputs=[
            gr.Textbox(
                label="Ask about Stephen's experience, skills, or background",
                placeholder="E.g., 'What are Stephen's key technical skills?'",
                lines=2
            )
        ],
        outputs=[
            gr.Textbox(
                label="AI Response",
                lines=10
            )
        ],
        title="Stephen's AI CV Assistant 🤖",
        description="""
        Welcome! Ask anything about Stephen's:
        • Technical skills
        • Work experience
        • Education
        • Projects
        • Book interests
        """,
        examples=[
            ["What is Stephen's current role and company?"],
            ["What are his key technical skills?"],
            ["What projects has he worked on?"],
            ["Can you give me a flavour of the books Stephen has read?"],
            ["What makes him a good data scientist?"]
        ],
        theme=gr.themes.Soft()
    )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860))
    )
