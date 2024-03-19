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
            # extract each cell value
            for cell in row:
                txt += str(cell.value)
        # try:
        # texts, nbPages = images_to_txt(pdf.read(), 'eng')
        # text_data_f = "\n\n".join(texts)
        # text+=text_data_f
        # except PDFPageCountError as e:
        # st.write(f".")
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


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask Questions from the PDF Files uploaded .. ✍️📝"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        new_db = FAISS.load_local("faiss_index", embeddings)  # Set allow_dangerous_deserialization=True
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        return response

    except Exception as e:
        st.error(f"Error occurred: {e}")


def main():
    st.set_page_config("Tanishq Ravula PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    with st.sidebar:
        st.image("Robot.png")
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ")

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  # user friendly message.
                raw_text = get_pdf_text(pdf_docs)  # get the pdf text
                text_chunks = get_text_chunks(raw_text)  # get the text chunks
                get_vector_store(text_chunks)  # create vector store
                st.success("Done")

        st.write("---")
        st.write("Tanishq Ravulas AI PDF Chatbot")  # add this line to display the image
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask Questions from the PDF Files uploaded .. ✍️📝"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                if response is not None:  # Check if response is not None
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                    st.markdown(full_response)  # Display full response
                    message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(message)
                else:
                    st.error("Failed to get a response. Please try again.")


if __name__ == "__main__":
    main()
