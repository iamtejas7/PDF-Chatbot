# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 7:07:39 2023

@author: Tejas
"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os

from dotenv import load_dotenv
load_dotenv()
google_api_key = os.environ.get('GOOGLE_API_KEY')


def read_pdf(pdf):
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()
        
    return text

def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 600,
        length_function = len
    )

    chunks = text_splitter.split_text(text=text)
    return chunks

def get_answer(vectorstore, query):
    docs = vectorstore.similarity_search(query=query,k=3)
    
    llm = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=google_api_key, convert_system_message_to_human=True, temperature=0)
    chain = load_qa_chain(llm=llm, chain_type= "stuff")

    response = chain.run(input_documents = docs, question = query)
    return response
    
    
with st.sidebar:
    st.title("PDF chatbot - powered by LangChain & Google's Gemini Pro")
    st.markdown('''
    Upload pdf and ask question related to content in pdf

    ## - Tejas Bankar

    - [Linkedin](https://www.linkedin.com/in/iamtejas7/)
    
    ''')


def main():
    st.header("Get Info from your pdf files by asking question to it", divider='rainbow')

    #upload a your pdf file
    pdf = st.file_uploader(":blue[Upload your PDF & wait till processing]", type='pdf')

    if pdf is not None:
        #st.write(pdf.name)
        
        # reading text from pdf
        text = read_pdf(pdf)
        
        # creating chunks of text
        chunks = create_chunks(text)

        # Loading Gemini Pro embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

        #Store the chunks in vector db
        vectorstore = Chroma.from_texts(chunks,embedding=embeddings)
        
        # Take User Question
        query = st.text_input("Ask question related to pdf file")
        st.write(f":blue[Input :] {query}")

        if query:
            if query.lower() == 'exit':
                vectorstore.delete_collection()
            else:
                # generating relevant answer
                response = get_answer(vectorstore, query)
                st.write(f":blue[Output :] {response}")


if __name__=="__main__":
    main()