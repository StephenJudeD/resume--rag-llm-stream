import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import logging
import os
import tempfile

# Required for AI model integration:
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.cloud import storage

#####################
# Logging Setup
#####################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#####################
# Google Credentials Setup
#####################
if os.getenv("GOOGLE_CREDENTIALS_JSON"):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp:
        temp.write(os.environ["GOOGLE_CREDENTIALS_JSON"])
        temp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp.name
        logger.debug(f"Temporary credentials file created at {temp.name}")
else:
    logger.error("GOOGLE_CREDENTIALS_JSON environment variable not set.")

#####################
# Utility: Download Vector Store Index Files from GCS
#####################
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
        if blob.name.endswith("/"):
            continue
            
        # Create local file path
        local_path = os.path.join(destination_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)
        logger.debug(f"Downloaded {blob.name} to {local_path}")

#####################
# Load FAISS Vector Store
#####################
def load_vector_store(embeddings):
    bucket_name = os.getenv("GCS_BUCKET_NAME", "ragsd-resume-bucket")
    index_path = os.getenv("GCS_INDEX_PATH", "faiss_indexes/cv_index_text-embedding-3-large")
    destination_folder = "/tmp/faiss_index"
    
    # Download all files in the index folder
    download_index_folder(bucket_name, index_path, destination_folder)
    
    # Debug directory contents
    contents = os.listdir(destination_folder)
    logger.debug(f"Index files downloaded: {contents}")
    
    try:
        vector_store = FAISS.load_local(
            destination_folder,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as error:
        logger.error("Error loading FAISS index. Verify downloaded files match FAISS requirements.")
        raise error

#####################
# Prompt Engineering & App Purpose
#####################
PROMPT_ENGINEERING = (
    "You are a precise CV analysis assistant. Your task is to:\n"
    "1. Only use information explicitly stated in the provided CV sections\n"
    "2. Quote specific details when possible\n"
    "3. If information is not found, clearly state 'Information not found in CV'\n"
    "4. Maintain chronological accuracy when discussing experience\n"
    "5. Consider all provided sections before answering\n"
    "6. Use relevant links of demos, where applicable, to emphasize skills"
)

#####################
# CVQueryApp with AI model integration
#####################
class CVQueryApp:
    def __init__(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set!")
            # Set the API key for the OpenAI library
            openai.api_key = api_key

            # Initialize OpenAIEmbeddings using the specified model
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=api_key
            )
            
            # Load the FAISS vector store (retrieval augmented generation)
            self.vector_store = load_vector_store(self.embeddings)
            logger.info("CVQueryApp initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing CVQueryApp: {e}")
            raise

    def query(self, question: str) -> str:
        try:
            docs = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7}
            ).get_relevant_documents(question)

            context = "\n".join(
                f"[{doc.metadata.get('section', 'N/A')}]\n{doc.page_content}"
                for doc in docs
            )

            system_msg = (
                PROMPT_ENGINEERING +
                "\nProvide an answer based solely on the following CV sections:\n" + context
            )

            response = openai.ChatCompletion.create( # Use openai.Completion.create
                model="gpt-4o-mini",  # Correct model name
                prompt=f"{system_msg}\nQuestion: {question}",  # Combine system and user message in prompt
                temperature=0.1,
                max_tokens=2000
            )

            return response.choices[0].text  # Access the text attribute

        except Exception as e:
            return f"Error: {str(e)}"

#####################
# StephenCVChatbot Class: UI & Callbacks
#####################
class StephenCVChatbot:
    def __init__(self, app, cv_app):
        self.app = app
        self.cv_app = cv_app
        self.layout = self._create_layout()
        self._create_callbacks()
        self._add_custom_css()

    def _create_layout(self):
        return html.Div([
            # Introductory Explanation Card
            dbc.Card(
                dbc.CardBody([
                    html.H4("Welcome to Stephen's CV Chat Assistant", className="card-title"),
                    html.P(
                        "This tool is designed for hiring managers to query Stephen’s professional background. "
                        "It is built on Stephen's CV and a curated list of books he has read in recent years. "
                        "Type your question below and press Enter or click 'Send'—the assistant will respond using "
                        "detailed prompt engineering instructions.",
                        className="card-text"
                    ),
                    html.P(
                        f"Prompt Engineering Instructions:<br><code>{PROMPT_ENGINEERING}</code>",
                        style={"fontSize": "0.9em", "color": "#555"}
                    )
                ]),
                className="mb-3"
            ),
            # Chat UI Title
            html.H2("Stephen's CV Chat Assistant", className="chat-title mb-3"),
            # Chat Card container
            dbc.Card([
                dbc.CardBody([
                    html.Div(id="chat-history", className="chat-history"),
                ], className="d-flex flex-column chat-body")
            ], className="chat-card mb-3"),
            # Chat Input Group
            dbc.InputGroup([
                dbc.Input(id="user-input", placeholder="Type your question here...", type="text"),
                dbc.Button("Send", id="send-button", color="success", n_clicks=0, className="ms-2"),
                dbc.Button("Clear", id="clear-button", color="danger", n_clicks=0, className="ms-2"),
            ], className="mb-3"),
            dcc.Store(id="chat-history-store"),
            dcc.Store(id="assistant-trigger"),
            html.Div(id="dummy-output", style={"display": "none"}),
        ], className="d-flex flex-column chat-container p-3")

    def _create_callbacks(self):
        # Callback when a user sends a message (via button click or pressing Enter)
        @self.app.callback(
            Output("chat-history-store", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Output("user-input", "value"),
            Output("assistant-trigger", "data"),
            Input("send-button", "n_clicks"),
            Input("user-input", "n_submit"),
            State("user-input", "value"),
            State("chat-history-store", "data"),
            prevent_initial_call=True
        )
        def update_chat(send_clicks, input_submit, user_input, chat_history):
            if not user_input:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": user_input})
            chat_display = self._format_chat_display(chat_history)
            # Append a typing indicator while waiting for the assistant response
            chat_display.append(self._create_typing_indicator())
            return chat_history, chat_display, "", {"trigger": True}

        # Callback to process the assistant's response
        @self.app.callback(
            Output("chat-history-store", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Input("assistant-trigger", "data"),
            State("chat-history-store", "data"),
            prevent_initial_call=True
        )
        def process_assistant_response(trigger, chat_history):
            if not trigger or not trigger.get("trigger"):
                return dash.no_update, dash.no_update

            chat_history = chat_history or []
            # Ensure that the last message is from the user
            if not chat_history or chat_history[-1]["role"] != "user":
                return dash.no_update, dash.no_update

            try:
                last_user_message = chat_history[-1]["content"]
                assistant_response = self.cv_app.query(last_user_message)
                chat_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
            except Exception as e:
                error_message = f"Error: {str(e)}"
                logger.error(error_message)
                chat_history.append({
                    "role": "assistant",
                    "content": error_message
                })
            chat_display = self._format_chat_display(chat_history)
            return chat_history, chat_display

        # Callback to clear the chat history
        @self.app.callback(
            Output("chat-history-store", "data", allow_duplicate=True),
            Output("chat-history", "children", allow_duplicate=True),
            Input("clear-button", "n_clicks"),
            prevent_initial_call=True
        )
        def clear_chat(n_clicks):
            if n_clicks:
                return [], []
            return dash.no_update, dash.no_update

    def _format_chat_display(self, chat_history):
        return [
            html.Div([
                html.Div(
                    msg["content"],
                    className=f"chat-message {msg['role']}-message"
                )
            ], className=f"message-container {msg['role']}-container")
            for msg in chat_history if isinstance(msg, dict) and "role" in msg
        ]

    def _create_typing_indicator(self):
        return html.Div([
            html.Div(className="chat-message assistant-message typing-message", children=[
                html.Div(className="typing-dot"),
                html.Div(className="typing-dot"),
                html.Div(className="typing-dot")
            ])
        ], className="message-container assistant-container")

    def _add_custom_css(self):
        custom_css = '''
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        body {
            font-family: 'DM Sans', sans-serif;
            background-color: #F9F7F4;
            margin: 0;
            padding: 0;
        }
        .chat-container {
            max-width: 800px;
            margin: 20px auto;
            background-color: #FFFFFF;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 90vh;
            display: flex;
            flex-direction: column;
        }
        .chat-title {
            font-size: 28px;
            font-weight: 700;
            color: #1B3139;
            text-align: center;
            margin-bottom: 10px;
        }
        .chat-card {
            border: none;
            background-color: #EEEDE9;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            margin-bottom: 20px;
            border-radius: 10px;
        }
        .chat-body {
            flex-grow: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .chat-history {
            flex-grow: 1;
            overflow-y: auto;
            padding: 15px;
        }
        .message-container {
            display: flex;
            margin-bottom: 15px;
        }
        .user-container {
            justify-content: flex-end;
        }
        .chat-message {
            max-width: 80%;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 16px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #004687;
            color: white;
        }
        .assistant-message {
            background-color: #7DC242;
            color: white;
        }
        .typing-message {
            background-color: #2D4550;
            color: #EEEDE9;
            display: flex;
            justify-content: center;
            align-items: center;
            min-width: 60px;
        }
        .typing-dot {
            width: 8px;
            height: 8px;
            background-color: #EEEDE9;
            border-radius: 50%;
            margin: 0 3px;
            animation: typing-animation 1.4s infinite ease-in-out;
        }
        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing-animation {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
        #user-input {
            border-radius: 20px;
            border: 1px solid #DCE0E2;
        }
        #send-button, #clear-button {
            border-radius: 20px;
            width: 100px;
        }
        #send-button {
            background-color: #7DC242;
            border-color: #7DC242;
        }
        #clear-button {
            background-color: #98102A;
            border-color: #98102A;
        }
        .input-group {
            flex-wrap: nowrap;
        }
        '''
        self.app.index_string = self.app.index_string.replace(
            "</head>",
            f"<style>{custom_css}</style></head>"
        )

        # Clientside callback: Auto-scroll chat history on update.
        self.app.clientside_callback(
            """
            function(children) {
                var chatHistory = document.getElementById('chat-history');
                if(chatHistory) {
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
                return '';
            }
            """,
            Output("dummy-output", "children"),
            Input("chat-history", "children"),
            prevent_initial_call=True
        )

#####################
# Main App Initialization
#####################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

cv_app = CVQueryApp()  # Initialize the AI model integration (vector store, embeddings, OpenAI client)
chatbot = StephenCVChatbot(app, cv_app)
app.layout = chatbot.layout

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run_server(host="0.0.0.0", port=port, debug=True)
