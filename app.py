import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import textwrap
from html_templates import css, user_template, bot_template
from langchain.vectorstores import FAISS
load_dotenv()

def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def get_pdf_text(pdf_docs):
    docs = []
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf.read())
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path)
        docs.extend(loader.load())
        os.remove(temp_file_path)
    return docs

def get_vectorstore(docs):
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, gemini_embeddings)
    
    return vector_store


def get_conversation_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Handle user input and display chat history
def handle_user_input(user_question):
    response = st.session_state.conversation.invoke(user_question)
    st.session_state.chat_history.append({'role': 'user', 'content': user_question})
    st.session_state.chat_history.append({'role': 'bot', 'content': wrap_text(response)})

    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AccountAssistant", page_icon="🏦")
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    st.header("Account Assistant")
    


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your Bank Statement as PDF and click Process", accept_multiple_files=True)
        
        if st.button("Process"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.vector_store = None

            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    docs = get_pdf_text(pdf_docs)
                    
                    vector_store = get_vectorstore(docs)
                    st.session_state.vector_store = vector_store
                    
                    st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                    st.success("Bank Statement processed successfully!")
            else:
                st.error("Please upload a Bank Statement PDF file.")

    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

    user_question = st.chat_input("Type your question here...", key="user_question")
    
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
