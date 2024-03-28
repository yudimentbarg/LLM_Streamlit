#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
from langchain.llms import Cohere
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS


# Set the API key for Cohere
os.environ["COHERE_API_KEY"] = "KJefdUzquaCHsNJ7SXgJFHZH3Ti132hyxmKMySGU"

# Initialize the language model
llm = Cohere()

# Define the directory path for document search
directory_path = r"C:\Users\yulia\Desktop\RAG BOOTCAMP\RAG-Bootcamp\source_documents"

# Use a text input field for the user to enter a query
query = st.text_input("Enter your query:", "", key="unique_query_input_key")

# Check if the query is not empty and has a reasonable length
if query and len(query.strip()) > 0:
    # Document search logic
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                          model_kwargs={'device':'cpu'},
                                          encode_kwargs={'normalize_embeddings': True})
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    docs = retriever.get_relevant_documents(query)

    # Display document search results
    for doc in docs:
        st.write(doc.page_content)

    # Web search logic (handled by the same Cohere model)
    try:
        web_search_results = llm(query)  # Make the API call
        st.write(f"Web Search Results: \n\n{web_search_results}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Please enter a more substantial query.")

