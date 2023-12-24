# PDF-Chatbot
PDF Q&amp;A chatbot using LangChain and Gemini Pro

# LangChain:
LangChain is a orchestration framework for LLMs. It give us whole flexibility to use LLMs and utilize its intelligence as per our usecase.

# Gemini Pro:
Google's Gemini Pro is a latest multi model which is capable on text as well as images. For now (as of Dec 2023) Gemini Pro API is available for free at limit of 60 calls per hour.

# Vector Database:
Vector Databases are kind of database or store which allow us to stor high dimensional vectors or embeddings of text/data. It has capability of semantic search, which means it can return similar vectors as an output. This are very important in GenAI applications and usecases like similarity seach, RAG.

# Streamlit:
Streamlit is an open source python library/framework used to create and share webpages/front end applications.

# General Pipeline of PDF-Chatbot :
1. Fetching text from uploaded pdf
2. Creating Chunks of extracted text
3. Applying Embeddings on chunks and storing it in Vector db
4. Taking user question
5. Applying same Embeddings on question text
6. Searching and extracting most similar chunks from vector db using (using semantic search or cosine similariy)
7. Passing most similar chunk along with question as an input prompt to LLM. (RAG)
8. LLM will generate relevant output as per user question.


