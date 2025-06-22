"""
The Main module of the Ollama RAG-based PDF chat application
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings

from rag.DocumentParser import DocumentParser
from rag.VectorStore import VectorStore
from rag.LLMManager import LLMManager
from rag.RAGPipeline import RAGPipeline


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

logger = logging.getLogger(__name__)


def extract_model_names(models_info):
    """Extract the model name"""
    try:
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {model_names}")
        return tuple()


def create_vector_db(file_upload, parser, vector_store):
    """Store the uploaded document into vectore DB"""

    logger.info(f"creating the vector db for the file uploaded: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())

    # parse the document and create chunks
    doc = parser.load_pdf(path)
    doc_chunks = parser.split_documents(doc)

    # use embeddings to save the document in vector store

    vector_db = vector_store.create_vector_db(documents=doc_chunks)
    logger.info(f"sucessfully added the document into vector db")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def process_query(question, rag_pipeline):
    """Invoke the Rag pipeline and answer the user query"""

    response = rag_pipeline.get_response(question)
    logger.info("Question processed and response generated")
    return response


def delete_vector_db(vector_db):
    """Delete the vector database and clear related session state"""
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            # Delete the collection
            vector_db.delete_collection()

            # Clear session state
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)

            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main():
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    # create the vector store and document parser obj
    vector_store = VectorStore()
    parser = DocumentParser()

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False

    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓",
            available_models,
            key="model_select",
        )

    # Add checkbox for sample PDF
    use_sample = col1.toggle("Use sample PDF", key="sample_checkbox")

    # when toggled, reset the values TODO: sample selection
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample

    if use_sample:
        sample_file_path = "../sample/bigData.pdf"
        if os.path.exists(sample_file_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("processing the PDF..."):
                    doc = parser.load_pdf(sample_file_path)
                    doc_chunks = parser.split_documents(doc)
                    st.session_state["vector_db"] = vector_store.create_vector_db(
                        documents=doc_chunks
                    )

                    # Open and display the sample PDF
                    with pdfplumber.open(sample_file_path) as pdf:
                        st.session_state["pdf_pages"] = [
                            page.to_image(resolution=300).original for page in pdf.pages
                        ]
        else:
            st.error(f"Sample PDF not found at: {sample_file_path}")
    else:
        file_upload = col1.file_uploader(
            "Upload a PDF file",
            type="pdf",
            accept_multiple_files=False,
            key="pdf_uploader",
        )

        if file_upload:
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDF..."):
                    st.session_state["vector_db"] = create_vector_db(
                        file_upload, parser, vector_store
                    )

                    st.session_state["file_upload"] = file_upload

                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"] = [
                            page.to_image(resolution=300).original for page in pdf.pages
                        ]

    # display pdf pages
    if st.session_state.get("pdf_pages"):
        zoom_level = col1.slider(
            "Zoom level",
            min_value=100,
            max_value=1000,
            value=600,
            step=50,
            key="zoom_slider",
        )
        
        # Add page selector for better navigation
        if len(st.session_state["pdf_pages"]) > 1:
            page_num = col1.selectbox(
                "Select page",
                range(1, len(st.session_state["pdf_pages"]) + 1),
                key="page_selector"
            )
            selected_page = page_num - 1
        else:
            selected_page = 0

        with col1:
            with st.container(height=410, border=True):
                if st.session_state["pdf_pages"]:
                    page_image = st.session_state["pdf_pages"][selected_page]
                    st.image(
                        page_image,
                        width=zoom_level,
                        caption=f"Page {selected_page + 1} of {len(st.session_state['pdf_pages'])}",
                        use_column_width=False
                    )

    # delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", type="secondary", key="delete_button"
    )

    if delete_collection:
        delete_vector_db(vector_db=st.session_state["vector_db"])

    # chat interface:
    with col2:
        if st.session_state["vector_db"] is not None:
            # Use the selected model instead of default
            llm_manager = LLMManager(model=selected_model)
            rag_pipeline = RAGPipeline(
                vector_db=st.session_state["vector_db"], llm_manager=llm_manager
            )
        else:
            st.info(
                "Please upload a PDF file or select the sample PDF to start chatting."
            )

        message_container = st.container(height=500, border=True)

        # display history messages
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # chat input and processing
        prompt = st.chat_input("Enter a prompt here...", key="chat_input")
        if prompt:
            try:
                st.session_state["messages"].append(
                    {"role": "user", "content": prompt}
                )
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # ask the LLM and retreive the response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_query(prompt, rag_pipeline)
                            st.markdown(response)
                        else:
                            st.warning("Please upload the PDF file first")
            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")


if __name__ == "__main__":
    main()
