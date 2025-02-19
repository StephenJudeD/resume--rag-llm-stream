import dash
from dash import html, dcc, Input, Output, State
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from google.cloud import storage
import logging

# Initialize logger (like in the first code)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def download_index(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.debug(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def load_vector_store(embeddings):
    """Loads vector store following the pattern from the first code"""
    bucket_name = "ragsd-resume-bucket"
    index_path = "faiss_indexes/cv_index_text-embedding-3-large"
    destination_folder = "/tmp"
    
    # Set credentials like in first code
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-credentials.json'
    
    # Download index
    destination_file_name = f"{destination_folder}/faiss_index"
    download_index(bucket_name, index_path, destination_file_name)
    
    return FAISS.load_local(
        destination_file_name,
        embeddings,
        allow_dangerous_deserialization=True
    )

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
            
            # Load vector store using the new pattern
            self.vector_store = load_vector_store(self.embeddings)
            
        except Exception as e:
            logger.error(f"Error initializing CVQueryApp: {str(e)}")
            raise

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

# Create Dash app
app = dash.Dash(__name__)
server = app.server

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Stephen's CV Chat Assistant</title>
        {%favicon%}
        {%css%}
        <style>
            .chat-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .chat-box {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                background: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .message-input {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            .submit-button {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .submit-button:hover {
                background-color: #0056b3;
            }
            .message {
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e3f2fd;
                margin-left: 20%;
            }
            .bot-message {
                background-color: #f5f5f5;
                margin-right: 20%;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize CVQueryApp
cv_app = CVQueryApp()

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("Stephen's CV Chat Assistant 🤖", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Div([
            html.P("Ask me anything about Stephen's experience, skills, or background!",
                   style={'textAlign': 'center', 'color': '#7f8c8d'}),
        ], className='chat-box'),
        
        html.Div(id='chat-history', className='chat-box'),
        
        dcc.Input(
            id='user-input',
            type='text',
            placeholder='Type your question here...',
            className='message-input'
        ),
        
        html.Button('Ask', id='submit-button', className='submit-button'),
        
        # Store component for chat history
        dcc.Store(id='chat-store', data=[]),
        
        # Example questions
        html.Div([
            html.H3("Example Questions:", style={'color': '#2c3e50'}),
            html.Ul([
                html.Li("What is Stephen's current role and company?"),
                html.Li("What are his key technical skills?"),
                html.Li("What projects has he worked on?"),
                html.Li("What books has Stephen read?"),
                html.Li("What makes him a good data scientist?"),
            ])
        ], className='chat-box')
    ], className='chat-container')
])

@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-store', 'data'),
     Output('user-input', 'value')],
    [Input('submit-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('chat-store', 'data')],
    prevent_initial_call=True
)
def update_chat(n_clicks, user_input, chat_history):
    if not user_input:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Get response from CVQueryApp
    response = cv_app.query(user_input)
    
    # Update chat history
    chat_history.append({'user': user_input, 'bot': response})
    
    # Create chat messages
    chat_messages = []
    for chat in chat_history:
        chat_messages.extend([
            html.Div(chat['user'], className='message user-message'),
            html.Div(chat['bot'], className='message bot-message')
        ])
    
    return chat_messages, chat_history, ''

if __name__ == '__main__':
    port = int(os.getenv("PORT", 7860))
    app.run_server(host='0.0.0.0', port=port, debug=True)
