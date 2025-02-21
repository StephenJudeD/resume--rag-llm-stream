Streamlit Meta App for Hiring Managers
======================================

Overview
--------

Welcome to Stephen's Meta App---an interactive platform designed for hiring managers to query my professional experience as a Data Scientist and Analyst. This Streamlit application utilizes advanced Retrieval-Augmented Generation (RAG) techniques to present insights gathered from my CV, Cover Letter, Dissertation, and Goodreads Book List, aiming to help evaluators understand my skills and qualifications more effectively.

Technologies Used
-----------------

-   Streamlit: For creating an interactive web application. https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps 
-   OpenAI: To leverage powerful language models for generating contextual responses based on user queries.
-   FAISS: A library for efficient similarity search and clustering of dense vectors, utilized to manage document embeddings.
-   Google Cloud Storage (GCS): For storing and managing the underlying vector data securely.
-   Python: As the primary programming language, enabling backend logic and application features.

Data Used
-----------------
-   CV
-   Cover Letter
-   List of book from Goodreads
-   Dissertation info

Installation Instructions
-------------------------

To set up this application locally, follow these steps:

1.  Clone this repository:

    ```
    git clone https://github.com/StephenJudeD/Resume-Rag.git
    cd Resume-Rag
    ```

2.  Create a virtual environment and activate it:

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required packages:

    ```
    pip install -r requirements.txt
    ```

4.  Set up environment variables for OpenAI and Google Cloud Storage access:

    ```
    1
    2
    3
    export OPENAI_API_KEY='your_openai_api_key'
    export GCS_BUCKET_NAME='your_bucket_name'
    export GCS_INDEX_PATH='your_index_path'
    ```

5.  Run the application:

    ```
    streamlit run app.py
    ```

Usage Examples
--------------

Once the application is running, users can interact with it through a chat interface. Here are some example queries you can ask:

-   "Can you tell me about Stephen's current role and his main responsibilities?"
-   "What technical skills has Stephen applied in his previous roles?"
-   "Can you describe some recent side projects Stephen has worked on?"
-   "Based on the books Stephen has read, what can you infer about his interests?"

The app will respond with pertinent insights derived from my extensive documentation.

Contribution Guidelines
-----------------------

I welcome contributions from the community! If you wish to contribute:

-   Fork the repository.
-   Create a new feature branch (`git checkout -b feature/new-feature`).
-   Make your changes and commit them (`git commit -m 'Add new feature'`).
-   Push to the branch (`git push origin feature/new-feature`).
-   Open a pull request to discuss your changes.

License
-------

This project is licensed under the MIT License. Feel free to use and modify it as per your requirements.

Contact Information
-------------------

For any inquiries, feel free to reach out:

-   Name: Stephen Donohoe
-   Email: stephenjudedon@gmail.com
-   GitHub: [github.com/StephenJudeD](https://github.com/StephenJudeD)
-   LinkedIn: [linkedin.com/in/stephen-donohoe-a0a02a192](https://www.linkedin.com/in/stephen-donohoe-a0a02a192)

* * * * *

Thank you for your interest in my application! I look forward to connecting.
