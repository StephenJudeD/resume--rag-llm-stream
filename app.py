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
            # Retrieve relevant documents using MMR
            docs = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
            ).get_relevant_documents(question)
    
            # Build context string from documents
            context = "\n".join(
                f"[{doc.metadata['section']}]\n{doc.page_content}" 
                for doc in docs
            )
    
            # Integrated system prompt with DeepSeek-style reasoning
            system_prompt = f"""You are a precise, pleasant hiring analysis assistant. Follow this process:
    
    <thinking>
    1. Determine if query is casual or requires document analysis
    2. For casual:
       - Respond playfully without document references
       - Keep under 2 sentences
    3. For analytical:
       a. Verify required info exists in [CV sections]
       b. Quote relevant details with section names
       c. Check chronological consistency
       d. Consider demo links from Projects
       e. If info missing: flag for clarification
    </thinking>
    
    Format responses as:
    <thinking>[Your structured reasoning]</thinking>
    <answer>[Final response with sign-off]</answer>
    
    Special Rules:
    1. USE ONLY information from these sections:
    {context}
    
    2. Mandatory responses:
       - Missing info: "I'm sorry, could you clarify?"
       - Stephen/water question: "Yes... according to Tinder 🌊"
       - Dublin weather: "Shite... but great for indoor ML! ☔"
    
    3. Always include:
       - Impeccable professional manners
       - 1:3 playful:professional tone ratio
       - "Anything else please do let me know 😊" at end
    
    4. Analysis requirements:
       → Cross-reference all sections first
       → Preserve chronological order
       → Highlight project demos when relevant
       → Explicitly note dated experiences"""
    
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {question}"}
            ]
    
            # Generate response using GPT-3.5 Turbo
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
    
            full_response = response.choices[0].message.content
    
            # Parse DeepSeek-style response
            thinking = ""
            answer = full_response  # Default if parsing fails
            
            try:
                if "<thinking>" in full_response and "</thinking>" in full_response:
                    thinking = full_response.split("<thinking>")[1].split("</thinking>")[0].strip()
                    answer_part = full_response.split("</thinking>")[1]
                    if "<answer>" in answer_part:
                        answer = answer_part.split("<answer>")[1].split("</answer>")[0].strip()
                    else:
                        answer = answer_part.strip()
            except Exception as parse_error:
                logger.warning(f"Response parsing error: {str(parse_error)}")
                answer = full_response  # Fallback to full response
    
            # Ensure mandatory sign-off
            if "😊" not in answer:
                answer += "\n\nAnything else please do let me know 😊"
    
            return answer, thinking
    
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return f"Apologies, I've encountered an error. Please try again later. 😊", ""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize app components
cv_app = CVQueryApp()

# Display title and app info
st.title("🤖 **Stephen-DS** _{AI Profile Explorer}_")
st.info("""
RAG-Powered Insights from CV, Cover Letter, Dissertation & Goodreads!  
Repository → [GitHub](https://github.com/StephenJudeD/resume--rag-llm-stream) 🔼
""")

# Sidebar control for clearing chat history
with st.sidebar:
    if st.button("🧹 Clear Chat History", help="Start a new conversation"):
        st.session_state.messages = []
        st.experimental_rerun()

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
    
    with st.spinner("🔍 Analyzing your question..."):
        promo, reasoning = cv_app.query(prompt)
    
    st.toast("✅ Response ready!", icon="🤖")
    
    # Append the promo (final answer) to chat history
    st.session_state.messages.append({"role": "assistant", "content": promo})
    
    # Optionally, display the chain-of-thought reasoning in an expander
    if reasoning:
        with st.expander("Show chain-of-thought reasoning"):
            st.markdown(reasoning)

# Display chat conversation from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
