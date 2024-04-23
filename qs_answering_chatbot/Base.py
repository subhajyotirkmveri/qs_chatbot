import streamlit as st
import os
import timeit
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

db_file_path='FAISS_Index'
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_path = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"

def creation_of_vectorDB_in_local(loader):
    data = loader.load()
    #db =FAISS.from_documents(data, embeddings)
    #db.save_local(db_file_path)    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    text = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(text, embeddings)
    db.save_local(db_file_path)    
    
def load_llm():
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm    

def creation_FAQ_chain():
    db=FAISS.load_local(db_file_path, embeddings)
    #retriever =db.as_retriever(score_threshold=0.7)
    retriever=db.as_retriever(search_kwargs={"k": 2})
    
    llm = load_llm()
    
    # Define the prompt template
    prompt_temp = """Use the following pieces of information to answer the user's question.
    Try to provide as much text as possible from "response". If you don't know the answer, please just say 
    "I don't know the answer". Don't try to make up an answer.
    
    Context: {context},
    Question: {question}
    
    Only return correct and helpful answer below and nothing else.
    Helpful answer: """

    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", 
                                        retriever=retriever, 
                                        input_key="query", 
                                        return_source_documents= True,
                                        chain_type_kwargs={"prompt" : PROMPT})
    return chain

