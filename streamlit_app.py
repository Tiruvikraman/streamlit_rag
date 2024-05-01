import streamlit as st

# Import your functions and modules here
#from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
'''
# Load the PDF document
pdf_docs = r"C:\Users\tiruv\Downloads\MF Data - March 2023 - April 2022.pdf"

def get_pdf_text():
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Get text chunks from the PDF
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
'''
# Load conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Load the vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
chain = get_conversational_chain()

# Streamlit app
def main():
    st.title("PDF Question Answering System")

    # User input
    user_question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if user_question:
            # Get relevant documents
            docs = new_db.similarity_search(user_question)
            # Generate response
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            # Display response
            st.write(response["output_text"])
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()

