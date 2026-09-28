import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="Legal Contract Simplifier", layout="centered")
st.title("⚖️ Legal Contract Simplifier (RAG)")
st.write("Upload a legal document (NDA / Rent Agreement) and ask questions.")

uploaded_file = st.sidebar.file_uploader("Upload Legal PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    You are a legal assistant. Answer the question ONLY using the provided context.
    If the answer is not in the context, say "This information is not present in the document."
    Do NOT use outside knowledge. Do NOT give legal advice.

    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    query = st.text_input("Ask a question about the contract:")

    if query:
        with st.spinner("Searching document..."):
            result = qa_chain({"query": query})
            st.success("Answer:")
            st.write(result["result"])

            with st.expander("📄 Source Chunks"):
                for i, doc in enumerate(result["source_documents"]):
                    st.write(f"**Chunk {i+1}:** {doc.page_content[:300]}...")