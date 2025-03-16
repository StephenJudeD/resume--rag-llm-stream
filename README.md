**Stephen's Meta App: AI-Powered Resume Explorer**
==================================================

**GitHub Repository:** [StephenJudeD/resume--rag-llm-stream](https://github.com/StephenJudeD/resume--rag-llm-stream)

* * * * *

**Overview**
------------

Welcome to **Stephen's Meta App**, an interactive, AI-powered platform designed for hiring managers to explore Stephen Donohoe's professional experience, skills, and qualifications. This application leverages cutting-edge **Retrieval-Augmented Generation (RAG)** techniques to provide contextual insights derived from Stephen's **CV**, **Cover Letter**, **Dissertation**, and **Goodreads Book List**. The app is built using **Streamlit**, offering a conversational interface for seamless interaction.

The app integrates **OpenAI's GPT-4o-mini** for natural language processing, **FAISS** for efficient similarity search, and **Google Cloud Storage (GCS)** for secure document storage. It employs advanced **LLM chaining** and **query optimization** techniques to deliver precise, context-aware responses.

* * * * *

**Technologies and Tools Used**
-------------------------------

### **Core Technologies**

1.  **Streamlit**:

    -   Used to create an interactive web application with a chat-based interface.
    -   Enables real-time interaction with the user and dynamic updates to the chat history.
    -   [Streamlit Documentation](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)
2.  **OpenAI**:

    -   Powers the language model (`gpt-4o-mini`) for generating contextual responses.
    -   Handles query optimization and response generation using **Chat Completions API**.
    -   [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/chat)
3.  **FAISS (Facebook AI Similarity Search)**:

    -   Manages dense vector embeddings for efficient similarity search and document retrieval.
    -   Enables the app to find relevant sections of Stephen's documents based on user queries.
    -   [FAISS Documentation](https://github.com/facebookresearch/faiss)
4.  **Google Cloud Storage (GCS)**:

    -   Stores and manages the FAISS index files securely in the cloud.
    -   Ensures scalability and accessibility of the underlying data.
    -   [GCS Documentation](https://cloud.google.com/storage/docs)
5.  **Python**:

    -   The primary programming language for backend logic, API integrations, and application features.
    -   Libraries used include `langchain`, `google-cloud-storage`, and `logging`.

* * * * *

**Data Sources**
----------------

The app extracts insights from the following documents:

1.  **CV**: Professional experience, skills, and responsibilities.
2.  **Cover Letter**: Personal motivations and career aspirations.
3.  **Dissertation**: Academic research and technical expertise.
4.  **Goodreads Book List**: Personal interests and intellectual pursuits.

* * * * *

**Pipeline and Workflow: In-Depth Breakdown**
=============================================

The **Pipeline and Workflow** section of the app is the backbone of its functionality. It connects various components---document embedding, query optimization, retrieval, response generation, and chat history management---to deliver a seamless, context-aware conversational experience. Below is a detailed breakdown of each step, including technical specifics, design choices, and rationale.

* * * * *

**1\. Document Embedding and Storage**
--------------------------------------

### **Embedding Generation**

-   **Process**:

    -   Documents (CV, Cover Letter, Dissertation, Goodreads Book List) are split into smaller chunks (e.g., paragraphs or sections).
    -   Each chunk is converted into a **dense vector embedding** using OpenAI's `text-embedding-3-large` model.
    -   These embeddings capture the **semantic meaning** of the text, enabling the app to understand and retrieve relevant information based on user queries.
-   **Why Use Embeddings?**

    -   Embeddings transform text into numerical representations that can be compared mathematically.
    -   This allows the app to find semantically similar sections of text, even if the exact keywords don't match.

### **FAISS Index Creation**

-   **Process**:

    -   The embeddings are indexed using **FAISS (Facebook AI Similarity Search)**, a library optimized for fast similarity search in high-dimensional spaces.
    -   The FAISS index is stored in **Google Cloud Storage (GCS)** for secure, scalable access.
-   **Why Use FAISS?**

    -   FAISS is highly efficient for **k-nearest neighbor (k-NN)** searches, making it ideal for retrieving the most relevant document sections.
    -   It supports **GPU acceleration** for even faster searches, though this app uses CPU-based indexing for simplicity.
-   **Why Store in GCS?**

    -   GCS provides secure, scalable storage for the FAISS index.
    -   It ensures the index is accessible from anywhere, making the app easy to deploy and maintain.

* * * * *

**2\. Query Optimization**
--------------------------

### **Pre-LLM Query Rewriting**

-   **Process**:

    -   User queries are rewritten using OpenAI's GPT-4o-mini to improve clarity, structure, and context.
    -   A **rewriting prompt** is used to guide the model:

        Copy

        ```
        "Rewrite the following query to be clear, structured, and detailed, as if it were being asked to a helpful assistant in a hiring-manager context. Provide any additional clarity or context if needed."

        ```

-   **Why Optimize Queries?**

    -   Raw user queries may be ambiguous or lack context.
    -   Rewriting ensures the query is clear and structured, improving the quality of the retrieved documents and the final response.
-   **Example**:

    -   **Original Query**: "Tell me about Stephen's skills."
    -   **Optimized Query**: "Can you provide a detailed overview of Stephen's technical and non-technical skills based on his CV and cover letter?"

* * * * *

**3\. Document Retrieval**
--------------------------

### **FAISS-Based Retrieval**

-   **Process**:

    -   The optimized query is converted into an embedding using the same `text-embedding-3-large` model.
    -   The embedding is used to search the FAISS index for the most relevant document sections.
    -   The app uses **Maximum Marginal Relevance (MMR)** to balance **relevance** and **diversity** in the retrieved results.
-   **What is MMR?**

    -   MMR is a retrieval strategy that selects documents that are both **relevant to the query** and **diverse from each other**.
    -   It prevents redundancy by ensuring the retrieved sections cover different aspects of the query.
-   **Why Use MMR?**

    -   Without MMR, the retrieved sections might be too similar, leading to repetitive or incomplete responses.
    -   MMR ensures the app provides a well-rounded set of information, improving the quality of the final response.
-   **Parameters**:

    -   `k`: Number of documents to retrieve (e.g., 8).
    -   `fetch_k`: Number of documents to consider before applying MMR (e.g., 20).
    -   `lambda_mult`: Controls the trade-off between relevance and diversity (e.g., 0.7).

* * * * *

**4\. Response Generation**
---------------------------

### **LLM Chaining**

-   **Process**:

    -   The retrieved document sections are passed as **context** to the GPT-4o-mini model.
    -   A **system prompt** guides the model to generate a concise, accurate, and context-aware response.
-   **System Prompt Components**:

    1.  **Persona**: "You are a concise, pleasant, and respectful analysis assistant for Hiring Managers to encourage them to hire Stephen."
    2.  **Constraints**:
        -   Use only the information in the provided sections.
        -   Quote specific details when possible.
        -   Include relevant demo links to emphasize skills.
    3.  **Chain-of-Thought (CoT) Instructions**:
        -   Provide 3 bullet points under "Reasoning:" detailing the thought process.
        -   Follow with a "Final Answer:" for the hiring manager.
    4.  **Easter Eggs**: Playful responses for specific queries (e.g., "Can Stephen walk on water?").
-   **Why Use CoT?**

    -   CoT improves transparency by showing the model's reasoning process.
    -   It helps users understand how the response was derived, building trust in the system.
-   **Example Response**:

    mipsasm

    Copy

    ```
    Reasoning:
    - Stephen's CV highlights his expertise in Python and SQL.
    - His dissertation demonstrates advanced data analysis skills.
    - The Goodreads book list suggests a strong interest in AI and technology.

    Final Answer: Stephen has extensive experience in Python, SQL, and data analysis, as evidenced by his CV and dissertation. His reading list also reflects a deep interest in AI and technology.

    ```

* * * * *

**5\. Chat History Management**
-------------------------------

### **Streamlit Session State**

-   **Process**:

    -   The app maintains a **chat history** using Streamlit's `session_state`.
    -   Each user query and assistant response is stored in `session_state.messages`.
-   **Why Use Streamlit Session State?**

    -   Streamlit's `session_state` is lightweight and easy to use for managing state in a web app.
    -   It allows the app to maintain context across interactions, enabling more natural conversations.
-   **Chat History vs. LangChain Memory**:

    -   **Streamlit Session State**:
        -   Simple and integrated directly into the app.
        -   Ideal for lightweight, short-term memory.
    -   **LangChain Memory**:
        -   More powerful and flexible, supporting long-term memory and complex workflows.
        -   Overkill for this app's use case, where short-term context is sufficient.
-   **Clearing History**:

    -   Users can clear the chat history to start a new conversation.
    -   This is implemented using a Streamlit button:

        python

        RunCopy

        ```
        if st.button("🔥 Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()

        ```

* * * * *

**Pipeline Integration**
------------------------

The pipeline is designed to be modular and efficient:

1.  **User Query** → 2. **Query Optimization** → 3. **Document Retrieval** → 4. **Response Generation** → 5. **Chat History Update**.

Each step is optimized for performance and accuracy, ensuring the app delivers high-quality, context-aware responses to hiring managers.

By combining **FAISS**, **OpenAI GPT-4o-mini**, and **Streamlit**, the app creates a powerful, user-friendly platform for exploring Stephen's professional profile.
* * * * *

**Key Features**
----------------

### **1\. Query Optimization**

-   **Purpose**: Improves the clarity and structure of user queries.
-   **Process**:
    -   A rewriting prompt is sent to GPT-4o-mini to refine the query.
    -   Example:
        -   **Original Query**: "Tell me about Stephen's skills."
        -   **Optimized Query**: "Can you provide a detailed overview of Stephen's technical and non-technical skills based on his CV and cover letter?"

### **2\. Chain-of-Thought Reasoning**

-   **Purpose**: Enhances transparency by detailing the model's thought process.
-   **Format**:

    Copy

    ```
    Reasoning:
    - Bullet point 1
    - Bullet point 2
    - Bullet point 3

    Final Answer: [Response]

    ```

### **3\. Easter Eggs**

-   **Purpose**: Adds a playful touch to the conversation.
-   **Examples**:
    -   **Query**: "Can Stephen walk on water?"
    -   **Response**: "Yes... according to Tinder."
    -   **Query**: "How's the weather in Dublin?"
    -   **Response**: "Shite..."

### **4\. Chat Interface**

-   **User Experience**:
    -   Users can ask questions in natural language.
    -   Responses are displayed in a conversational format.
    -   Example Queries:
        -   "What are Stephen's main responsibilities in his current role?"
        -   "What technical skills has Stephen applied in previous roles?"
        -   "What can you infer about Stephen's interests from his Goodreads book list?"

* * * * *

**Installation and Setup**
--------------------------

### **Prerequisites**

-   Python 3.8+
-   OpenAI API Key
-   Google Cloud Storage Bucket

### **Steps**

1.  Clone the repository:

    bash

    Copy

    ```
    git clone https://github.com/StephenJudeD/resume--rag-llm-stream.git
    cd resume--rag-llm-stream

    ```

2.  Create a virtual environment:

    bash

    Copy

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    ```

3.  Install dependencies:

    bash

    Copy

    ```
    pip install -r requirements.txt

    ```

4.  Set environment variables:

    bash

    Copy

    ```
    export OPENAI_API_KEY='your_openai_api_key'
    export GCS_BUCKET_NAME='your_bucket_name'
    export GCS_INDEX_PATH='your_index_path'

    ```

5.  Run the app:

    bash

    Copy

    ```
    streamlit run app.py

    ```

* * * * *

**Contributing**
----------------

Contributions are welcome! Follow these steps:

1.  Fork the repository.
2.  Create a new feature branch:

    bash

    Copy

    ```
    git checkout -b feature/new-feature

    ```

3.  Commit your changes:

    bash

    Copy

    ```
    git commit -m 'Add new feature'

    ```

4.  Push to the branch:

    bash

    Copy

    ```
    git push origin feature/new-feature

    ```

5.  Open a pull request.

* * * * *

**License**
-----------

This project is licensed under the **MIT License**.

* * * * *

**Contact Information**
-----------------------

-   **Name**: Stephen Donohoe
-   **Email**: <stephenjudedon@gmail.com>
-   **GitHub**: [StephenJudeD](https://github.com/StephenJudeD)
-   **LinkedIn**: [Stephen Donohoe](https://www.linkedin.com/in/stephen-donohoe-a0a02a192)

* * * * *

Thank you for exploring Stephen's Meta App! 🚀
