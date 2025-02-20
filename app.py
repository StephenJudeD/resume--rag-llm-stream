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
    index_path = os.getenv("GCS_INDEX_PATH", "faiss_indexes/cv_index_text-embedding-3-large")
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

# Initialize the app
cv_app = CVQueryApp()

# Custom styling
import streamlit as st

# Configure app page
st.set_page_config(
    page_title="Stephen's Professional Profile Assistant",
    page_icon="📄",
    layout="wide"
)

## Custom styling
st.set_page_config(page_title="Stephen's Professional Profile Assistant", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
        /* General styling */
        body {
            background-color: #f0f4f8;
            font-family: 'Inter', sans-serif;
        }
        .stTextInput>div>input {
            padding: 0.75rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            background-color: white;
            font-size: 1rem;
            width: 100%;
        }
        .stTextInput>div>input:focus {
            outline-color: #2563eb;
        }
        /* Chat bubbles */
        .message-container {
            display: flex;
            gap: 12px;
            margin-bottom: 1rem;
        }
        .user-message-container {
            justify-content: flex-end;
        }
        .message-bubble {
            max-width: 70%;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            line-height: 1.4;
            font-size: 0.875rem;
        }
        .user-bubble {
            background: #2563eb;
            color: white;
            align-self: flex-end;
        }
        .bot-bubble {
            background: #e2e8f0;
            color: #1e293b;
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            padding: 1rem;
            background-color: #fff;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        /* Chat container */
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
            padding: 1rem;
            background-color: #fff;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar for personal info
with st.sidebar:
    st.image("https://via.placeholder.com/150", caption="Your Photo", use_column_width=True)  # Replace with your image URL
    st.markdown("### Stephen Doe")
    st.markdown("""
        **Data Scientist | AI Engineer**  
        Passionate about building intelligent systems and solving complex problems.  
        [LinkedIn](#) | [GitHub](#) | [Portfolio](#)
    """)

# Main content
st.title("Stephen's Professional Profile Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat container
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.markdown(f"<div class='message-container user-message-container'><div class='message-bubble user-bubble'>{chat['user']}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='message-container'><div class='message-bubble bot-bubble'>{chat['bot']}</div></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# User input section
user_input = st.text_input("Ask about experience, skills, or projects...", key="user_input", on_change=lambda: st.session_state.update({"run_query": True}))

# Example questions for quick selection
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Current role and company?"):
        user_input = "Current role and company?"
        st.session_state.run_query = True
with col2:
    if st.button("Technical skills?"):
        user_input = "Technical skills?"
        st.session_state.run_query = True
with col3:
    if st.button("Recent projects?"):
        user_input = "Recent projects?"
        st.session_state.run_query = True
with col4:
    if st.button("Book recommendations?"):
        user_input = "Book recommendations?"
        st.session_state.run_query = True

# Auto-run query on Enter or button click
if st.session_state.get("run_query", False) and user_input:
    response = cv_app.query(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})
    st.session_state.run_query = False
    st.experimental_rerun()
