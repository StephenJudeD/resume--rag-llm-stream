import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import os
import tempfile
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.cloud import storage
import logging

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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Stephen's CV Chat Assistant</title>
        {%favicon%}
        {%css%}
        <style>
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
                display: flex;
                flex-direction: column;
                padding: 20px;
                height: 90vh;
            }
            .chat-title {
                font-size: 28px;
                font-weight: 700;
                color: #1B3139;
                text-align: center;
                margin-bottom: 10px;
            }
            .info-card {
                margin-bottom: 20px;
            }
            .example-card {
                margin-bottom: 20px;
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
            .chat-history {
                flex-grow: 1;
                overflow-y: scroll;
                padding: 15px;
                min-height: 0;
                max-height: 50vh;
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
            .bot-message {
                background-color: #7DC242;
                color: white;
            }
            #user-input {
                border-radius: 20px;
                border: 1px solid #DCE0E2;
                flex-grow: 1;
            }
            .send-button {
                background-color: #7DC242;
                border-color: #7DC242;
                border-radius: 20px;
                width: 100px;
                margin-left: 10px;
            }
            .clear-button {
                background-color: #98102A;
                border-color: #98102A;
                border-radius: 20px;
                width: 100px;
                margin-left: 10px;
            }
            .input-group {
                flex-wrap: nowrap;
            }
            .example-button {
                margin: 5px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

cv_app = CVQueryApp()

app.layout = html.Div(
    className="chat-container",
    children=[
        html.H2("Stephen's CV Chat Assistant", className="chat-title"),
        dbc.Card(
            dbc.CardBody(
                html.P(
                    "Welcome to Stephen's CV Chat Assistant! This tool allows hiring managers to query Stephen’s professional background "
                    "and his reading interests. It is built on Stephen’s CV and a curated list of books he has read in recent years.",
                    className="card-text"
                )
            ),
            className="info-card"
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Example Queries", className="card-title"),
                    dbc.Button("What's Stephen's recent work experience?",
                               id="example-1",
                               color="primary",
                               className="example-button"),
                    dbc.Button("What books has Stephen read recently?",
                               id="example-2",
                               color="info",
                               className="example-button"),
                    dbc.Button("How does Stephen describe his technical skills?",
                               id="example-3",
                               color="secondary",
                               className="example-button"),
                ]
            ),
            className="example-card"
        ),
        dbc.Card(
            className="chat-card",
            children=[
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-chat",
                        type="circle",
                        children=html.Div(
                            id="chat-history",
                            className="chat-history"
                        )
                    ),
                ])
            ]
        ),
        dbc.InputGroup([
            dbc.Input(
                id="user-input",
                placeholder="Type your question here...",
                type="text",
                n_submit=0  # trigger on enter key
            ),
            dbc.Button(
                "Send",
                id="submit-button",
                n_clicks=0,
                className="send-button"
            ),
            dbc.Button(
                "Clear",
                id="clear-button",
                n_clicks=0,
                className="clear-button"
            ),
        ]),
        dcc.Store(id="chat-store", data=[]),
        html.Div(id="dummy-output", style={"display": "none"}),
    ]
)

# Main chat callback: listens to both the Send button and the Enter (n_submit) event
@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-store', 'data'),
     Output('user-input', 'value')],
    [Input('submit-button', 'n_clicks'),
     Input('user-input', 'n_submit'),
     Input('clear-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('chat-store', 'data')],
    prevent_initial_call=True
)
def update_chat(send_clicks, enter_submit, clear_clicks, user_input, chat_history):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'clear-button':
        return [], [], ''
    
    if not user_input:
        return dash.no_update, dash.no_update, dash.no_update

    response = cv_app.query(user_input)
    chat_history.append({'user': user_input, 'bot': response})
    
    chat_messages = []
    for msg in chat_history:
        chat_messages.extend([
            html.Div(
                className="message-container user-container",
                children=[
                    html.Div(
                        msg['user'],
                        className="chat-message user-message"
                    )
                ]
            ),
            html.Div(
                className="message-container",
                children=[
                    html.Div(
                        msg['bot'],
                        className="chat-message bot-message"
                    )
                ]
            )
        ])
    
    return chat_messages, chat_history, ''

# Callback for example queries
@app.callback(
    Output('user-input', 'value'),
    [
        Input('example-1', 'n_clicks'),
        Input('example-2', 'n_clicks'),
        Input('example-3', 'n_clicks')
    ],
    prevent_initial_call=True
)
def update_example_query(n1, n2, n3):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'example-1':
        return "What's Stephen's recent work experience?"
    elif trigger_id == 'example-2':
        return "What books has Stephen read recently?"
    elif trigger_id == 'example-3':
        return "How does Stephen describe his technical skills?"
    return dash.no_update

if __name__ == '__main__':
    port = int(os.getenv("PORT", 7860))
    app.run_server(host='0.0.0.0', port=port, debug=True)
