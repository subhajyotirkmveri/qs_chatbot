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

DATA_PATH = r"/home/sysadm/Downloads/LLM/data"
DB_FAISS_PATH = "db_faiss_index"
model_path = "models/llama-2-7b-chat.ggmlv3.q8_0.bin"

def create_vector_db(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    text = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(text, embeddings)
    db.save_local(DB_FAISS_PATH)

# Define the prompt template
prompt_template = """Use the following pieces of information to answer the user's question.
    Try to provide as much text as possible from "response". If you don't know the answer, please just say 
    "I don't know the answer". Don't try to make up an answer.
    
    Context: {context},
    Question: {question}
    
    Only return correct and helpful answer below and nothing else.
    Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=db.as_retriever(search_kwargs={"k": 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": prompt})
    return qa_chain

def qa_chat():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(file_path, query):
    create_vector_db(file_path)
    qa_result = qa_chat()
    response = qa_result({"query": query})
    return response

# Main Streamlit app code
def main():
    st.title("Q&A Chatbot with PDF Support")

    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_file:
        
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            
            f.write(uploaded_file.getbuffer())
        st.success("PDF file uploaded successfully!")

        query = st.text_input("Enter your question:")
        if query:
            start_time = timeit.default_timer()
            result = final_result(file_path, query)
            end_time = timeit.default_timer()
            output = result.get("result") if result else None
            
            Source_documents = result.get("source_documents") if result else None

            # Display the result and execution time
            if output:
                st.write("Result:", output)
                st.write('=' * 50)
                st.write("Source Documents:", Source_documents)
            else:
                st.write("No answer found.")
                
            st.write('=' * 50)
            st.write("Time to retrieve answer:", end_time - start_time, "sec")

if __name__ == "__main__":
    main()

