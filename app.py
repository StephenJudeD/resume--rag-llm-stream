import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.cloud import storage
import os
import tempfile
import logging

# Configure logging
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

# Function to download files from Google Cloud Storage
def download_index_folder(bucket_name, source_folder, destination_dir):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    os.makedirs(destination_dir, exist_ok=True)
    blobs = bucket.list_blobs(prefix=source_folder)
    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        local_path = os.path.join(destination_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)
        logger.debug(f"Downloaded {blob.name} to {local_path}")

# Load the vector store
def load_vector_store(embeddings):
    bucket_name = os.getenv("GCS_BUCKET_NAME", "ragsd-resume-bucket")
    index_path = os.getenv("GCS_INDEX_PATH", "faiss_indexes/cv_index_text-embedding-3-large_v2")
    destination_folder = "/tmp/faiss_index"
    download_index_folder(bucket_name, index_path, destination_folder)
    contents = os.listdir(destination_folder)
    logger.debug(f"Index files downloaded: {contents}")
    try:
        vector_store = FAISS.load_local(destination_folder, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as error:
        logger.error("Error loading index. Verify downloaded files match FAISS requirements.")
        raise error

# Define CVQueryApp class
class CVQueryApp:
    def __init__(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found!")
            self.client = OpenAI(api_key=api_key)
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
            self.vector_store = load_vector_store(self.embeddings)
        except Exception as e:
            logger.error(f"Error initializing CVQueryApp: {str(e)}")
            raise

    def query(self, question: str) -> str:
        try:
            docs = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}).get_relevant_documents(question)
            context = "\n".join(f"[{doc.metadata['section']}]\n{doc.page_content}" for doc in docs)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are a precise & Pleasant CV analysis assistant, you're purpose is to speak to Hiring Managers. Your task is to:\n"
                        "1. Only use information explicitly stated in the provided CV sections\n"
                        "2. Quote specific details when possible\n"
                        "3. If information is not found, clearly state 'I am sorry, I didn't quite get that, can you please clarify'\n"
                        "4. Maintain chronological accuracy when discussing experience\n"
                        "5. Consider all provided sections before answering\n"
                        "6. Use relevant links of demoes, where relevant, to emphasise skills\n"
                        "7. Reply in a professionl & playful manner\n"
                        "8. Small talk & Pleasantires are permitted, you can indulge\n"
                        "9. Sign-off every response when asked a CV related {context} with 'Anything else please do let me know 😊'\n"
                        "10. Start every response with 'Thanks for asking, I would be happy to respond'\n"
                        "11. When discussing books, start with genre, flavour, then give actually book examples"
                    )},
                    {"role": "user", "content": f"Based on these CV sections:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

import streamlit as st

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize app components (ensure CVQueryApp is imported/defined)
cv_app = CVQueryApp()

# Display title and app info
st.title("🤖 **Stephen-DS** _{AI Profile Explorer}_")
st.info("""
RAG-Powered Insights from CV, Cover Letter, Dissertation & Goodreads! Code repository → [GitHub](https://github.com/StephenJudeD/resume--rag-llm-stream) 🔼
""")

# Sidebar control for clearing chat history
with st.sidebar:
    if st.button("🧹 Clear Chat History", help="Start a new conversation"):
        st.session_state.messages = []
        st.rerun()

    # ideas for users here
    st.markdown("### Ideas to Ask")
    st.markdown("""
    - "What does Stephen's Goodreads book list reveal about his personal interests and continual learning?"
    - "Tell me more about Stephen's recent side projects and their implemetation"
    - "Are there any recurring themes that indicate what drives his professional passion?"
    - "Can you infer how Stephen’s hobbies and reading choices align with his professional skills?"
    - "What are the main research contributions highlighted in Stephen's dissertation?
    """)

# Handle user query
if prompt := st.chat_input("Ask about my experience, skills, projects, or books..."):
    # Append user's message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process the query without immediate inline rendering of messages
    with st.spinner("🔍 Analyzing your question..."):
        response = cv_app.query(prompt)
    st.toast("✅ Response ready!", icon="🤖")
    
    # Append assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat conversation from history only once
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
