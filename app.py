import gradio as gr
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.cloud import storage
import logging
import tempfile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Write temporary credentials file from env var
if os.getenv("GOOGLE_CREDENTIALS_JSON"):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
        temp.write(os.environ["GOOGLE_CREDENTIALS_JSON"])
        temp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp.name
        logger.debug(f"Temporary credentials file created at {temp.name}")
else:
    logger.error("GOOGLE_CREDENTIALS_JSON environment variable not set.")




def download_index_folder(bucket_name, source_folder, destination_dir):
    """Download all files in a GCS folder to a local directory"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # List all blobs in the folder
    blobs = bucket.list_blobs(prefix=source_folder)
    
    for blob in blobs:
        # Skip directories (they have trailing '/')
        if blob.name.endswith('/'):
            continue
            
        # Create local file path
        local_path = os.path.join(destination_dir, os.path.basename(blob.name))
        
        # Download the file
        blob.download_to_filename(local_path)
        logger.debug(f"Downloaded {blob.name} to {local_path}")

def load_vector_store(embeddings):
    bucket_name = os.getenv("GCS_BUCKET_NAME", "ragsd-resume-bucket")
    index_path = os.getenv("GCS_INDEX_PATH", "faiss_indexes/cv_index_text-embedding-3-large")
    destination_folder = "/tmp/faiss_index"
    
    # Download all files in the index folder
    download_index_folder(bucket_name, index_path, destination_folder)
    
    # Debug directory contents
    contents = os.listdir(destination_folder)
    logger.debug(f"Index files downloaded: {contents}")
    
    # Load FAISS from the directory
    try:
        vector_store = FAISS.load_local(
            destination_folder,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as error:
        logger.error("Error loading index. Verify downloaded files match FAISS requirements.")
        raise error

class CVQueryApp:
    def __init__(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found!")
    
            self.client = OpenAI(api_key=api_key)
    
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=api_key
            )
    
            self.vector_store = load_vector_store(self.embeddings)
    
        except Exception as e:
            logger.error(f"Error initializing CVQueryApp: {str(e)}")
            raise
    
    def query(self, question: str) -> str:
        try:
            docs = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
            ).get_relevant_documents(question)
    
            context = "\n".join(f"[{doc.metadata['section']}]\n{doc.page_content}" for doc in docs)
    
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are a precise CV analysis assistant. Your task is to:\n"
                        "1. Only use information explicitly stated in the provided CV sections\n"
                        "2. Quote specific details when possible\n"
                        "3. If information is not found, clearly state 'Information not found in CV'\n"
                        "4. Maintain chronological accuracy when discussing experience\n"
                        "5. Consider all provided sections before answering\n"
                        "6. Use relevant links of demos, where applicable, to emphasize skills"
                    )},
                    {"role": "user", "content": f"Based on these CV sections:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
    
        except Exception as e:
            return f"Error: {str(e)}"

def create_gradio_app(cv_app):
    # Custom CSS to match your Dash styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        max-width: 800px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px !important;
        border-radius: 5px !important;
        margin: 5px 0 !important;
    }
    .user-message {
        background-color: #e3f2fd !important;
        margin-left: 20% !important;
    }
    .assistant-message {
        background-color: #f5f5f5 !important;
        margin-right: 20% !important;
    }
    .title {
        text-align: center !important;
        color: #2c3e50 !important;
    }
    """

    def chat_response(message, history):
        """Handle chat interactions"""
        return cv_app.query(message)

    # Create the Gradio interface
    with gr.Blocks(css=custom_css) as demo:
        gr.HTML("<h1 class='title'>Stephen's CV Chat Assistant 🤖</h1>")
        
        chatbot = gr.Chatbot(
            label="Chat History",
            bubble_full_width=False,
            height=400
        )
        
        msg = gr.Textbox(
            label="Ask me anything about Stephen's experience, skills, or background!",
            placeholder="Type your question here...",
            scale=4
        )
        
        with gr.Row():
            submit = gr.Button("Ask", variant="primary")

        # Example questions
        gr.HTML("""
            <div style='padding: 20px; background: white; border-radius: 10px; margin-top: 20px;'>
                <h3 style='color: #2c3e50;'>Example Questions:</h3>
                <ul>
                    <li>What is Stephen's current role and company?</li>
                    <li>What are his key technical skills?</li>
                    <li>What projects has he worked on?</li>
                    <li>What books has Stephen read?</li>
                    <li>What makes him a good data scientist?</li>
                </ul>
            </div>
        """)

        # Set up chat functionality
        msg.submit(
            chat_response, 
            [msg, chatbot], 
            [chatbot]
        ).then(
            lambda: "", 
            None, 
            [msg]
        )
        
        submit.click(
            chat_response, 
            [msg, chatbot], 
            [chatbot]
        ).then(
            lambda: "", 
            None, 
            [msg]
        )

    return demo

# Initialize and launch
if __name__ == "__main__":
    cv_app = CVQueryApp()
    demo = create_gradio_app(cv_app)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False
    )
