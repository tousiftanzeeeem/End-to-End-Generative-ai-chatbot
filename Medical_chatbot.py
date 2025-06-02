from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.2
)


# loader=PyPDFLoader("book.pdf")
# docs=loader.load()


# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
# chunks=text_splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# 3. Create FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
# 4. Use FAISS as a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt = PromptTemplate(
    input_variables=["context","question"],
    template="""
    You are a medical expert.
    You are given a question and you need to answer it based on the {context} you have given.
    If there is not enough information provided just say "I Have no knowledge  about it"
    Question: {question}
    Answer:
    """
)
parser=StrOutputParser()

chain=prompt|llm|parser
# Correct usage — function is called properly


st.title('Ai Medical Chatbot')
question=st.text_input("Ask me anything about medical")
submit=st.button("Submit")
if question and submit:
    context = retriever.invoke(question)
    context_text="".join(doc.page_content for doc in context)
    result = chain.invoke({
        "context":context_text,
        "question":question
    })
    st.write(result)