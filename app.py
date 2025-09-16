import os
import warnings
import logging
import streamlit as st
import dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


dotenv.load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("📄 Multi-PDF Chatbot with Groq (PDF-Only Answers)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Step 1: Multiple PDF upload
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Embedding model
   
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    # Temporary vectorstore in memory
    vectorstore = None
    docs = []

    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(uploaded_file.name)
        docs.extend(loader.load())

    # Split and embed all PDFs together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(texts, embedding)

    # Step 2: User question
    prompt = st.chat_input("Ask something about your PDFs")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        model = "llama-3.1-8b-instant"
        groq_chat = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name=model
        )

        # Custom strict prompt
        custom_prompt = PromptTemplate(
            template="""You are a helpful assistant.
Answer the following question strictly using the provided context.
If the context does not contain the answer, reply with "I don't know".

Context:
{context}

Question:
{question}

Answer:""",
            input_variables=["context", "question"]
        )

        # RetrievalQA chain with custom prompt
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True
        )

        try:
            result = chain({"query": prompt})
            response = result["result"]

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Show sources
            if result.get("source_documents"):
                with st.expander("📂 Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'PDF')}")
                        st.write(doc.page_content[:300] + "...")
        except Exception as e:
            st.error(f"Error: {str(e)}")
