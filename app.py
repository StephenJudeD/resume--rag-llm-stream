import gradio as gr
import os
import asyncio
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.cloud import storage
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Google Cloud credentials
if os.getenv("GOOGLE_CREDENTIALS_JSON"):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
        temp.write(os.environ["GOOGLE_CREDENTIALS_JSON"])
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp.name
        logger.info("GCS credentials initialized")
else:
    logger.error("Missing GOOGLE_CREDENTIALS_JSON")

class CVQueryApp:
    def __init__(self):
        self.initialized = False
        self.initialization_lock = asyncio.Lock()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        self.vector_store = None

    async def initialize(self):
        if self.initialized:
            return
            
        async with self.initialization_lock:
            try:
                logger.info("Starting async initialization...")
                
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY missing!")
                    
                self.client = AsyncOpenAI(api_key=self.api_key)
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=self.api_key
                )
                
                await self.load_vector_store()
                self.initialized = True
                logger.info("Initialization completed successfully")
                
            except Exception as e:
                logger.error(f"Initialization failed: {str(e)}")
                raise

    async def load_vector_store(self):
        bucket_name = os.getenv("GCS_BUCKET_NAME", "ragsd-resume-bucket")
        index_path = os.getenv("GCS_INDEX_PATH", "faiss_indexes/cv_index_text-embedding-3-large")
        destination_folder = "/tmp/faiss_index"
        
        await self.download_index_folder(bucket_name, index_path, destination_folder)
        
        try:
            loop = asyncio.get_event_loop()
            self.vector_store = await loop.run_in_executor(
                None,
                lambda: FAISS.load_local(
                    destination_folder,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            )
        except Exception as e:
            logger.error("Vector store loading failed")
            raise

    async def download_index_folder(self, bucket_name, source_folder, destination_dir):
        from concurrent.futures import ThreadPoolExecutor
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        def sync_download():
            os.makedirs(destination_dir, exist_ok=True)
            blobs = bucket.list_blobs(prefix=source_folder)
            
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue
                local_path = os.path.join(destination_dir, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
                logger.info(f"Downloaded {blob.name}")

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, sync_download)

    async def query(self, question: str) -> str:
        try:
            if not self.initialized:
                await self.initialize()
                
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
            )
            
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None,
                lambda: retriever.get_relevant_documents(question)
            )
            
            context = "\n".join(
                f"[{doc.metadata['section']}]\n{doc.page_content}" 
                for doc in docs
            )
            
            response = await self.client.chat.completions.create(
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
            
        except asyncio.TimeoutError:
            logger.warning("Query timeout")
            return "Response timeout - please try a more specific question"
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return f"Error processing request: {str(e)}"

# Initialize application
cv_app = CVQueryApp()

# Create Gradio interface
with gr.Blocks(title="CV Assistant", theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center'>Stephen's CV Chat Assistant 🤖</h1>")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your Question")
    clear = gr.Button("Clear")
    
    msg.submit(cv_app.query, msg, chatbot)
    clear.click(lambda: None, None, chatbot)

app = demo.app

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860))
    )
