import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pandas as pd
import openpyxl
from pytesseract import Output, TesseractError
from functions import convert_pdf_to_txt_pages, convert_pdf_to_txt_file, save_pages, displayPDF, images_to_txt
import fitz  # PyMuPDF
import pdf2image
from pdf2image.exceptions import PDFPageCountError
import google.generativeai as genai
import asyncio
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(csvexcelattachment):
    txt = ""
    file_name = csvexcelattachment.name
    file_extension = file_name.split('.')[-1].lower()

    if file_extension == 'csv':
        df = pd.read_csv(csvexcelattachment)
        txt += '   Dataframe: \n' + df.to_string()
    elif file_extension in ['xlsx', 'xls']:
        wb = openpyxl.load_workbook(csvexcelattachment)
        sheet = wb.active
        for row in sheet.iter_rows():
            for cell in row:
                txt += str(cell.value)
    return txt


def get_text_chunks(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, and elaborate on every point.
    Provide examples, explanations, and any relevant information. If the answer is not in the provided context, just say, "answer is not available in the context".
    Do not provide incorrect information.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.9)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask Questions from the CSV and Excel Files uploaded .. ✍️📝"}]


async def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            return "No relevant documents found."

        chain = get_conversational_chain()

        response = chain.invoke(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        if not response or 'output_text' not in response:
            return "No valid response generated."

        return response['output_text']

    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None


def main():
    st.set_page_config("Tanishq Ravula Large Csv and Excle Chatbot", page_icon=":scroll:")
    st.header("CSV and EXCEL 📚 - Chat Agent 🤖 ")

    with st.sidebar:
        st.image("Robot.png")
        st.write("---")
        st.title("📁 CSV and Excel File's Section")
        pdf_docs = st.file_uploader("Upload your CSV or Excel Files & \n Click on the Submit & Process Button ",type=["csv","xlsx","xls"])

        if st.button("Submit & Process"):
            with st.spinner("Processing..."): 
                raw_text = get_pdf_text(pdf_docs)  
                text_chunks = get_text_chunks(raw_text)  
                get_vector_store(text_chunks)  
                st.success("Done")

        st.write("---")
        st.write("Tanishq Ravulas AI CSV and EXCEL Chatbot")  
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask Questions from the CSV and Excel Files uploaded .. ✍️📝"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(user_input(prompt))
                if response:
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)
                else:
                    st.write("No valid response generated.")


if __name__ == "__main__":
    main()
