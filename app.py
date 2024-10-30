import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS    #vector embeddings provided by Meta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain   #helps in QA chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv



load_dotenv()

#configuring api key for gemini model through env 
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

#to load pdf and read text
def get_pdf(pdf_docs):
    text= ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


#chunking the retrieved text into small pieces of data with chunk size of 1000 words and 200 with overlap for context relevancy.
def text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks= text_splitter.split_text(text)
    return chunks


#after tokenization/chunking, we convert to vector embeddings.
def get_vector_store(text_chunks):
    #vector embedding technique from google AI completely free for everyone.
    embeddings =GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    
    #save vector store locally using save_local function.
    vector_store.save_local("faiss_index")

def conversational_chain():
    prompt_template= """
    Answer the prompt as accurately as possible from the provided context with all the desired details required for better understanding.
    Incase if the answer is not possible, try saying "The information you desired is beyond my context understanding." and try to avoid
    incorrect answers.\n\n
    Context: \n {context}?\n
    Question: \n{question}\n

    Answer: 
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt= PromptTemplate(template=prompt_template, input_variables=['context','question'])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)   #stuff is data type for document chains.
    return chain

def user_input(user_question, context):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



#main function for streamlit interface
def main():
    st.set_page_config("Chat PDF")
    st.header("ChatPDF Application 💁")

    if "context" not in st.session_state:
        st.session_state.context = ""

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question,st.session_state.context)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf(pdf_docs)
                text_chunk = text_chunks(raw_text)
                get_vector_store(text_chunk)
                st.success("Done")

    if st.session_state.context:
        st.subheader("Conversation History")
        st.text(st.session_state.context)

if __name__ == "__main__":
    main()
 