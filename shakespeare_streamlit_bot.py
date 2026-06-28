import os
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title='Shakespeare', layout='wide', initial_sidebar_state="expanded")
st.title("📜 Words with William 🖋️")

st.markdown("""
*Ask questions about any of Shakespeare's 37 plays, 154 sonnets, 
and narrative poems. Powered by RAG over the complete Folger Shakespeare corpus.*
""")

with st.expander("📚 Works in the knowledge base"):
    st.markdown("""
    **Tragedies:** Hamlet, Macbeth, Othello, King Lear, Romeo & Juliet...
    **Comedies:** A Midsummer Night's Dream, Much Ado About Nothing...
    **Histories:** Henry V, Richard III, Henry IV...
    **Poetry:** Sonnets, Venus and Adonis, Lucrece
    """)

load_dotenv()

# API key — Anthropic only
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("🔑 ANTHROPIC_API_KEY not found. Add it to your .env file.")
    st.stop()

# Sidebar
st.sidebar.image('.streamlit/will_shakespeare.png', use_column_width=True)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
k = st.sidebar.slider("Docs to retrieve", 1, 10, 4)
max_tokens = st.sidebar.slider("Max tokens", 64, 2048, 512, step=64)

# Load vectorstore
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory=os.getenv("VECTORSTORE_PATH", "./shakespeare_vectorstore"),
    embedding_function=emb
)

# Build retrieval chain
system_prompt = ChatPromptTemplate.from_template("""You are a Shakespeare scholar. 
Answer questions using only the context from Shakespeare's works provided below.
If the question is unrelated to Shakespeare, politely redirect.

Context: {context}
Question: {input}
Answer:""")

llm = ChatAnthropic(
    temperature=temperature,
    max_tokens=max_tokens,
    model="claude-haiku-4-5",
    api_key=api_key
)

combine_docs_chain = create_stuff_documents_chain(llm, system_prompt)
qa = create_retrieval_chain(
    vectordb.as_retriever(search_kwargs={"k": k}),
    combine_docs_chain
)

# iOS-style CSS
st.markdown("""
<style>
  [data-testid="stChatMessage"][role="user"] .stMarkdown {
    background-color: #0b93f6!important; color:white!important;
    border-radius:18px 18px 4px 18px!important;
  }
  [data-testid="stChatMessage"][role="assistant"] .stMarkdown {
    background-color: #e5e5ea!important; color:#000!important;
    border-radius:18px 18px 18px 4px!important;
  }
  [data-testid="stChatMessage"] .stMarkdown {
    padding:8px 12px!important; margin:4px 0!important;
  }
</style>
""", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

avatar_img = Image.open('.streamlit/will_shakespeare.png')

for msg in st.session_state.messages:
    avatar = avatar_img if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle input
if user_input := st.chat_input("Ask me about Shakespeare…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        result = qa.invoke({"input": user_input})
        response = result["answer"]
    except Exception as e:
        response = f"⚠️ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar=avatar_img):
        st.markdown(response)