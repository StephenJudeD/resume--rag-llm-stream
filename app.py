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
        """Processes a question and returns an answer based on the CV content"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.get_relevant_documents(question)

            # Format context
            context = "\nRELEVANT CV SECTIONS:\n\n"
            for doc in retrieved_docs:
                context += f"[{doc.metadata['section']}]\n{doc.page_content}\n\n"

            # Prepare messages for GPT
            messages = [
                {"role": "system", "content": """You are a precise & Pleasant CV analysis assistant, you're purpose is to speak to Hiring Managers. Your task is to:
                    1. Only use information explicitly stated in the provided CV sections
                    2. Quote specific details when possible
                    3. If information is not found, clearly state 'I am sorry, I didn't quite get that, can you please clarify'
                    4. Maintain chronological accuracy when discussing experience
                    5. Consider all provided sections before answering
                    6. Use relevant links of demoes, where relevant, to emphasise skills
                    7. Reply in a professionl & playful manner
                    8. Small talk & Pleasantires are permitted, you can indulge
                    9. Sign-off every response with 'Anything else please do let me know 😊'
                    10. Start every response with 'Thanks for asking, I would be happy to respond'
                    10. When discussing books, start with genre, flavour, then give actually book examples"""},
                {"role": "user", "content": f"Based on these CV sections:\n{context}\n\nQuestion: {question}"}
            ]

            # Get response from GPT
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=2000
            )

# Initialize the app
cv_app = CVQueryApp()

import streamlit as st

# Title
st.title("👨‍💻 Stephen's Meta Profile")
st.info(
    "Retrieval-Augmented Generation (RAG) Insights gathered from my CV, Cover Letter, Dissertation, and Goodreads Book List. The code used, and further information, can be found @ [GitHub](https://github.com/StephenJudeD/resume--rag-llm-stream/blob/main/README.md)"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "run_query" not in st.session_state:
    st.session_state.run_query = False

# Chat container
with st.container():
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            st.write(message["bot"])

# Input container
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Ask about my experience, skills, projects, or books that I have read...", key="text_input")
with col2:
    if st.button("Ask") and user_input:
        st.session_state.user_input = user_input
        st.session_state.run_query = True

# Quick Questions
with st.expander("Quick Questions"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Can you tell me about Stephen's current role and how long, in years, he has worked there?"):
            st.session_state.user_input = "Can you tell me about Stephen's current role and how long, in years, he has worked there?"
            st.session_state.run_query = True

        if st.button("Can you describe some of the technical skills Stephen has and how he applied them in previous roles?"):
            st.session_state.user_input = "Can you describe some of the technical skills Stephen has and how he applied them in previous roles?"
            st.session_state.run_query = True

    with col2:
        if st.button("Can you tell me about some recent side projects Stephen has worked on and what they entailed?"):
            st.session_state.user_input = "Can you tell me about some recent side projects Stephen has worked on and what they entailed?"
            st.session_state.run_query = True

        if st.button("Can you tell me some books that Stephen has read?"):
            st.session_state.user_input = "Can you tell me some books that Stephen has read?"
            st.session_state.run_query = True

# Process user input
if st.session_state.run_query and st.session_state.user_input:
    response = cv_app.query(st.session_state.user_input)
    st.session_state.chat_history.append({"user": st.session_state.user_input, "bot": response})
    st.session_state.user_input = ""
    st.session_state.run_query = False
    st.experimental_rerun()
