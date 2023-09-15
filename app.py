import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import time
# extract data from all the documents
# docs is a list of all the documents
def get_data(docs):
    data = ""
    for doc in docs:
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            data += page.extract_text()
    return data


#convert data to chunks
def get_chunks(raw_data):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(raw_data)
    return chunks

#convert chunks to embeddings and create vector store
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

#ask question about document and follow up questions
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # maintains a conversation history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

#handles the user input
def handle_user_input(user_input):
    #generates the final answer in JSON format
    with st.spinner("Analyzing..."):
        response = st.session_state.conversation({'question':user_input})
        with st.chat_message("assistant"):
           st.write(response["answer"])

def main():
    load_dotenv()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY")=="":
        st.write("OPENAI_API_KEY is not set. Please add your key in .env file.")
        exit(1)

    st.set_page_config(page_title="Knowledge Base Chatbot", page_icon="💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Knowledge Base Chatbot 💬")

    with st.chat_message("assistant"):
        st.write("Hello👋, How can I help you today?")

    user_input = st.chat_input("Ask your query")        
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        handle_user_input(user_input)

        
    #add a sidebar for uploading of user documents
    with st.sidebar:
        st.subheader("Your documents")
        
        #docs -> list of all documents
        docs = st.file_uploader("Upload your docs here and click on 'Process'",accept_multiple_files=True)
        if docs == []:
            st.warning("Please upload documents.")
        elif docs and st.button("Process"):
            with st.spinner("Processing..."):
                #get data from the document
                raw_data = get_data(docs)

                #divide data into chunks
                chunks = get_chunks(raw_data)

                #convert to embeddings and create vector store
                vectorstore = get_vectorstore(chunks)

                if vectorstore:
                    st.caption("Processing Completed!")
                    
                #create conversation chain (user input)
                st.session_state.conversation = get_conversation_chain(vectorstore)
    
                       
if __name__=='__main__':
    main()
