import streamlit as st
import cohere
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

# Load the API key securely
api_key = st.secrets["KJefdUzquaCHsNJ7SXgJFHZH3Ti132hyxmKMySGU"]

# Initialize the Cohere client
co = cohere.Client(api_key)

# Define the directory path for document search
directory_path = r"C:\Users\yulia\Desktop\RAG BOOTCAMP\RAG-Bootcamp\source_documents"

# Use a text input field for the user to enter a query
query = st.text_input("Enter your query:", "", key="unique_query_input_key")

# Add a button to trigger the process
if st.button("Process Query"):
    if query and len(query.strip()) > 0:
        # Document search logic
        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                              model_kwargs={'device': 'cpu'},
                                              encode_kwargs={'normalize_embeddings': True})
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        docs = retriever.get_relevant_documents(query)

        # Display document search results
        for doc in docs:
            st.write(doc.page_content)

        # Web search logic (handled by the same Cohere model)
        try:
            web_search_results = co.generate(prompt=query, max_tokens=50).generations[0].text  # Make the API call
            st.write(f"Web Search Results: \n\n{web_search_results}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a more substantial query.")



