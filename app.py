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
    #index_path = os.getenv("GCS_INDEX_PATH", "faiss_indexes/cv_index_text-embedding-3-large_v2")
    index_path = os.getenv("GCS_INDEX_PATH", "faiss_indexes/cv_index_text-embedding-v2")
    
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
            # Optionally, you could use a different embedding model name
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
            self.vector_store = load_vector_store(self.embeddings)
        except Exception as e:
            logger.error(f"Error initializing CVQueryApp: {str(e)}")
            raise

    def query(self, question: str) -> (str, str):
        try:
            docs = self.vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
            ).get_relevant_documents(question)
            
            context = "\n".join(
                f"[{doc.metadata['section']}]\n{doc.page_content}" for doc in docs
            )
            # Updated system prompt includes clear chain of thought (CoT) instructions.
            system_prompt = (
                "You are a concise, pleasant, and respectful analysis assistant for Hiring Managers to encourage them to hire Stephen, providing detailed answers where necessary. "
                "Your task is to analyze the provided CV sections using the following instructions:\n\n"
                "1. Use only the information explicitly given in the provided sections.\n"
                "2. Quote specific details when possible.\n"
                "3. If information is missing, clearly state: 'I am sorry, I didn't quite get that, can you please clarify?'\n"
                "4. Keep your answer chronologically accurate.\n"
                "5. Consider all the provided sections before answering.\n"
                "6. When appropriate, include relevant demo links to emphasize skills.\n"
                "7. Use impeccable manners. Small talk and pleasantries are permitted in a playful tone.\n"
                "8. Include a pleasant sign-off based on the answer provided, where relevant, to encourage further examination of the previous query.\n"
                "9. If the user asks 'Can Stephen walk on water?' - reply 'Yes... according to Tinder'\n"
                "10. If the user asks 'Hows the weather in Dublin' - reply 'Shite...'\n"
                "11. **Chain-of-thought instructions:** First, provide 3 rich and concise data bullet points under 'Reasoning:' detailing your thought process. "
                "Then, after a clear marker, provide your 'Final Answer:' for the hiring manager.\n\n"
                "Please format your reply as follows:\n\n"
                "Reasoning:\n- Bullet point 1\n- Bullet point 2\n- Bullet point 3\n\n"
                "Final Answer: [Your final answer here]\n"
            )
            
            # In your query method's messages construction:
            messages = [
                {"role": "system", "content": system_prompt},
                *st.session_state.get('messages', [])[-4:],  # Last 2 exchanges
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
]
            
            # NOTE: gpt-4o-mini or gpt-3.5-turbo
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=3000
            )
            
            full_response = response.choices[0].message.content
            
            # Attempt to split the answer into promo (final answer) and reasoning parts
            if "Final Answer:" in full_response:
                reasoning_part, promo_part = full_response.split("Final Answer:", 1)
                reasoning_part = reasoning_part.replace("Reasoning:", "").strip()
                promo_part = promo_part.strip()
            else:
                # Fallback if markers aren't detected.
                reasoning_part = ""
                promo_part = full_response
            
            return promo_part, reasoning_part
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return f"Error: {str(e)}", ""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize app components
cv_app = CVQueryApp()

# Display title and app info
st.title("🤖 **Stephen-DS** _{AI Profile Explorer}_")
st.info("""
Explore Stephen's profile through AI-powered insights. **Start chatting now!** 💬

RAG-Powered Insights from CV, Cover Letter, Dissertation & Goodreads!
Repository → [GitHub](https://github.com/StephenJudeD/resume--rag-llm-stream) 🚀
""")

# Sidebar control for clearing chat history
with st.sidebar:
    if st.button("🧹 Clear Chat History", help="Start a new conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### Ideas to Ask")
    st.markdown("""
    - "Can you infer an overview of technical and non-technical skills relating to his most recent role?"
    - "What does Stephen's Goodreads book list reveal about his personal interests?"
    - "How's the weather in Dublin"
    - "Tell me about recent side projects and their implementation"
    - "Are there recurring themes that indicate what drives his professional passion?"
    - "Can Stephen walk on water?"
    """)

# Handle user query
if prompt := st.chat_input("Ask about my experience, skills, projects, or books..."):
    # Append user's message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("🪇 Analyzing your question..."):
        promo, reasoning = cv_app.query(prompt)
    
    st.toast("😎 Response ready!", icon="🤖")
    
    # Append the promo (final answer) to chat history
    st.session_state.messages.append({"role": "assistant", "content": promo})
    
    # display the chain-of-thought reasoning in an expander
    if reasoning:
        with st.expander("Show chain-of-thought reasoning"):
            st.markdown(reasoning)

# Display chat conversation from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
