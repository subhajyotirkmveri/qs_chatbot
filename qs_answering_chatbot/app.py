from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import tempfile
import streamlit as st
from Base import creation_FAQ_chain, creation_of_vectorDB_in_local

def pdf_loader(tmp_file_path):
    loader = PyPDFLoader(file_path=tmp_file_path)
    return loader

def main():
    st.set_page_config(page_title="QA Chatbot", page_icon="😈", layout="wide")
    st.title("Q&A Chatbot with PDF Support 📃")

    with st.sidebar:
        st.title("Settings")
        st.markdown('---')
        st.subheader('Upload Your PDF File')
        doc = st.file_uploader("Upload your PDF file and Click Process", 'pdf')

        if st.button("Process"):
            with st.spinner("Processing"):
                if doc is not None:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(doc.read())  # Use read() instead of getvalue()
                        tmp_file_path = tmp_file.name
            
                    st.success(f'File {doc.name} is successfully saved!')
                    
                    load = pdf_loader(tmp_file_path)
                    creation_of_vectorDB_in_local(load)
                    st.success("Process Done")
                else:
                    st.error("❗️Please Upload Your File❗️")
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"**User:** {message['content']}")
        elif message["role"] == "assistant":
            st.markdown(f"**Assistant:** {message['content']}")

    query = st.text_input("Ask the Question")
    if st.button("Submit"):
        if query:
            ans = creation_FAQ_chain()
            result = ans(query)
        #a = result["result"]
            output = result.get("result") if result else None
            if output:
                st.write("Result:", output)
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": output})

if __name__ == '__main__':
    main()


