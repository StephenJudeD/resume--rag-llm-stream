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
