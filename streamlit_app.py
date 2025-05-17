import streamlit as st
import time
from io import StringIO
import os
import fitz
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf(data):
    doc = fitz.Document(stream = data)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = load_pdf(os.path.join(folder_path, filename))
            documents.append({"filename": filename, "text": text})
    return documents

class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=st.secrets["API_KEY"]
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": st.secrets["API_KEY"]}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        openai_api_key = openai_api_key or st.secrets["API_KEY"]
        super().__init__(base_url=st.secrets["BASE_URL"], openai_api_key=openai_api_key, **kwargs)

class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query, k=3):
        D, I = self.index.search(query, k)  # D: distances, I: indices
        results = []
        for idx in I[0]:
            results.append(self.metadata[idx])
        return results

embed_model_id = 'intfloat/e5-small-v2'
model_kwargs = {"device": "cpu", "trust_remote_code": True}

def create_index(documents):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    texts = [doc["text"] for doc in documents]
    metadata = [{"filename": doc["filename"], "text": doc["text"]} for doc in documents]

    # Generate embeddings
    embeddings_matrix = [embeddings.embed_query(text) for text in texts]
    embeddings_matrix = np.array(embeddings_matrix).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    # Return a FAISSIndex object that contains both the index and metadata
    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    results = faiss_index.similarity_search(query_embedding, k=k)
    return results

template = ""

selected_model = "minstralai/minstral-7b-instruct:free"
model = ChatOpenRouter(model_name = selected_model)

def answer_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

if "query" not in st.session_state:
    st.session_state.query = ""

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "files" not in st.session_state:
    st.session_state.files = []

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.session_state.files.append(uploaded_file)

    for file in st.session_state.files:
        # To read file as bytes:
        bytes_data = file.getvalue()
        if file.type == "application/pdf":
            string_data = load_pdf(bytes_data)
        else:
            stringio = StringIO(file.getvalue().decode("utf-8"))
            string_data = stringio.read()

        st.write(string_data)

    if st.button("Clear"):
        st.session_state.files = []
        uploaded_file = None

st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

st.caption("Note that this demo app isn't actually connected to any LLMs. Those are expensive ;)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = model.chat.completions.create(model = st.secrets["MODEL"], messages = st.session_state.messages)
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.choices[0].message.content.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})