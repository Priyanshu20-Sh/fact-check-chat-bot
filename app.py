import streamlit as st
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

try:
    from transformers.utils import logging
    logging.disable_progress_bar()
except ImportError:
    pass

import uuid
from dotenv import load_dotenv
load_dotenv()

# Set up page configurations at the very beginning
st.set_page_config(
    page_title="Autonomus Rag",


    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the core router logic
import Self_agent_rag.router as router
from langchain_core.messages import HumanMessage

from document_loader.load import load_document
from document_loader.semantic import semantic
from vector_store.hybrid_bm_faiss import hybrid_docs
from langchain_core.documents import Document

# Custom premium CSS
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None
if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = None

# Sidebar File Uploader
with st.sidebar:
    st.title("Document Source")
    uploaded_file = st.file_uploader("Upload PDF or Word Document", type=["pdf", "docx"])
    
    if uploaded_file:
        file_name = uploaded_file.name
        
        # Avoid redundant re-indexing if the file remains unchanged
        if st.session_state.indexed_file != file_name:
            with st.spinner("Processing..."):
                try:
                    # Load file to docs
                    docs = load_document(uploaded_file)
                    if isinstance(docs, str):
                        docs = [Document(page_content=docs)]
                    
                    # Compute semantic splits and build the index
                    list_chunks = semantic(docs)
                    new_hybrid = hybrid_docs(list_chunks)
                    
                    router.hybrid_retriever = new_hybrid
                    st.session_state.hybrid_retriever = new_hybrid
                    
                    st.session_state.indexed_file = file_name
                    st.success("Indexed successfully")
                except Exception as e:
                    st.session_state.indexed_file = file_name
                    st.error(f"Error: {e}")
        else:
            st.info(f"Active Document: {file_name}")
            if st.session_state.hybrid_retriever is not None:
                router.hybrid_retriever = st.session_state.hybrid_retriever

# Main Page Header
st.title("Fact Checker Agent")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Enter your query..."):
    # Display user query
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Run agent loop
    with st.spinner("Thinking..."):
        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = router.app.invoke(
                {
                    "question": prompt,
                    "messages": [HumanMessage(content=prompt)],
                    "retry_count": 0
                },
                config=config
            )
            
            # Extract answer
            final_answer = result.get("final_answer") or result.get("generate_answer") or "No response generated."
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(final_answer)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer
            })
            
        except Exception as e:
            st.error(f"Error: {e}")
