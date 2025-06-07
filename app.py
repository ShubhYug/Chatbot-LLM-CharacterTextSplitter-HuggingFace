import streamlit as st
from backend import get_qa_chain


# App title
st.set_page_config(page_title="CodersDaily Course Assistant 💬")
st.title("🎓 Ask About CodersDaily Courses")


# Load QA chain only once
@st.cache_resource
def load_chain():
    return get_qa_chain()


qa_chain = load_chain()


# Input box
query = st.text_input("Ask a question about our Data Science or Analytics courses:")


# Handle input
if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
    st.markdown(f"**💬 Answer:** {answer}")