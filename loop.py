import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

st.title("PDF Summarizer, QA & Chat")


if "messages" not in st.session_state:
    st.session_state.messages = []


if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def load_openai_api_key():
    dotenv_path = "openai.env"
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(f"Unable to retrieve OPENAI_API_KEY from {dotenv_path}")
    return openai_api_key


def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

# interaction
def handle_user_interaction(pdf_files, user_question):
    content_found = False

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        knowledgeBase = process_text(text)


        docs = knowledgeBase.similarity_search(user_question)
        if docs:
            OpenAIModel = "gpt-3.5-turbo"
            llm = ChatOpenAI(model=OpenAIModel, temperature=0.1, openai_api_key=load_openai_api_key())
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=user_question)

            if response:
                content_found = True
                st.session_state.messages.append({"role": "user", "content": user_question})
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Answer for {user_question} in {pdf.name}:\n\n{response}"})


    if not content_found:
        st.write("I couldn't find any information about your question in the uploaded PDFs.")



if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar.expander("Upload your PDF Documents"):
    pdf_files = st.sidebar.file_uploader(' ', type='pdf', accept_multiple_files=True)
    if pdf_files:
        st.session_state.uploaded_files = pdf_files

#  history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    pdf_files = st.session_state.uploaded_files
    user_question = prompt.strip()

    if pdf_files and user_question:
        st.session_state.messages = [message for message in st.session_state.messages if message["role"] != "assistant"]

#  user question and PDF
        handle_user_interaction(pdf_files, user_question)

        for message in st.session_state.messages:
            if message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
