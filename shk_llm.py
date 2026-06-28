# Requisite imports
import os, glob
import gradio as gr
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# imports for langchain, plotly, and Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_chroma import Chroma
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE

# Let's implement the model I'm using
model = "gpt-4o-mini"
db_name = "vector_db"

# It's bad practice to show the environmental variables, but I do want to acknowledge and load them here

load_dotenv()

os.listdir("/Users/toelou/Desktop/shakespeare_data")

# Note, apparently the following are also considered to fall under the "romance" category
# pericles, cymbeline, the tempest, the two noble kinsmen

genre_map = {
    # comedies
    "merry_wives_of_windsor.pdf": "comedy",
    "a_midsummer_nights_dream.pdf": "comedy",
    "twelfth_night_or_what_you_will.pdf": "comedy",
    "pericles.pdf":"comedy",
    "much_ado_about_nothing.pdf":"comedy",
    "taming_of_the_shrew.pdf":"comedy",
    "two_gentlemen_of_verona.pdf":"comedy",
    "cymbeline.pdf":"comedy",
    "as_you_like_it.pdf":"comedy",
    "loves_labors_lost.pdf":"comedy",
    "the_tempest.pdf":"comedy",
    "measure_for_measure.pdf":"comedy",
    "the_two_noble_kinsmen.pdf":"comedy",
    "the_winters_tale.pdf":"comedy",
    "alls_well_that_ends_well.pdf":"comedy",
    "the_comedy_of_errors.pdf":"comedy",
    # histories
    "henry_v.pdf":"history",
    "henry_iv_pt1.pdf":"history",
    "henry_iv_pt2.pdf":"history",
    "henry_vi_pt1.pdf":"history",
    "henry_vi_pt2.pdf":"history",
    "henry_vi_pt3.pdf":"history",
    "richard_ii.pdf": "history",
    "richard_iii.pdf": "history",
    "king_john.pdf": "history",
    "henry_viii.pdf": "history",
    # tragedies
    "othello.pdf": "tragedy",
    "coriolanus.pdf": "tragedy",
    "julius_caesar.pdf": "tragedy",
    "king_lear.pdf": "tragedy",
    "romeo_and_juliet.pdf": "tragedy",
    "hamlet.pdf": "tragedy",
    "antony_and_cleopatra.pdf": "tragedy",
    "macbeth.pdf": "tragedy",
    "troilus_and_cressida.pdf": "tragedy",
    "timon_of_athens.pdf": "tragedy",
    "titus_andronicus.pdf": "tragedy",
    # poetry
    "shakespeares_sonnets.pdf":"poetry", 
    "lucrece.pdf":"poetry",
    "venus_and_adonis.pdf":"poetry",
    "phoenix_and_turtle.pdf":"poetry"
}

# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase
# Where all the data comes from: https://www.folger.edu/explore/shakespeares-works/download/

# Path connected to that directory containing Shakespeare's data
folder = "/Users/toelou/Desktop/shakespeare_data"
docs = []

for filename in os.listdir(folder):
    if filename.endswith(".pdf"):
        path = os.path.join(folder, filename)
        loader = PyPDFLoader(path)
        loaded_docs = loader.load()

        genre = genre_map.get(filename, "unknown")
        for doc in loaded_docs:
            doc.metadata["genre"] = genre

        docs.extend(loaded_docs)

genres = [doc.metadata.get("genre", "unknown") for doc in docs]

splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 250)
chunks = splitter.split_documents(docs)

len(chunks)

# Put the newly generated chunks into a Vector Store that associates the vector embeddings with each chunk

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_name = "shakespeare_vectorstore"

# Delete existing DB if it exists
if os.path.exists(db_name):
    Chroma(persist_directory = db_name, embedding_function = embeddings).delete_collection()

# Build a vectorstore 
vectorstore = Chroma.from_documents(documents = chunks, 
                                    embedding = embeddings, 
                                    persist_directory = db_name,
                                    collection_name = "shakespeare_collection")

print(f"This is a newly created vectorstore with {vectorstore._collection.count()} documents in it.")

# See the dimensions in a particular vector

collection = vectorstore._collection
sample_embedding = collection.get(limit = 5, include = ["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

result = collection.get(include = ['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['genre'] for metadata in result['metadatas']]
colors = [['blue','green','red', 'yellow', 'purple'][['poetry','comedy','tragedy', 'history', 'unknown'].index(t)] for t in doc_types]

tsne = TSNE(n_components=2, random_state = 42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data = [go.Scatter(
    x = reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

# Implemnting this in 3D

tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

# Gradio Chatbot

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

# Time to build a new chat with OpenAI
# llm = ChatAnthropic(temperature=0.8, model="claude-haiku-4-5")

# Establish the conversation memory for the chat
# memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)

# The retriever is an abstraction over the VectorStore that will be used during RAG
# retriever = vectorstore.as_retriever()

# Put it all together: set up the conversation chain with the GPT 4o-mini LLM, the vector store, and memory
# conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = retriever, memory = memory)

# query = "Can you tell me something about Romeo and Juliet?"
# result = conversation_chain.invoke({"question": query})
# print(result["answer"])

# Set up a new conversation memory for the chat
# memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
# conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = retriever, memory = memory)

# Quick and easy prototype on a chat with an LLM
# Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain

# def chat(message, history):
#     try:
#         result = conversation_chain.invoke({"question": message})
#         answer = result.get("answer", "🤖 No response generated.")
#         if not isinstance(answer, str):
#             answer = str(answer)
#         return answer
#     except Exception as e:
#         return f"⚠️ Error: {str(e)}"

# Let's take a look at this in Gradio

# view = gr.ChatInterface(chat).launch(inbrowser = True)