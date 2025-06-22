"""
This is a class handling the document parser.
"""
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentParser:
    """Handles PDF document loading and processing."""
    def __init__(self, chunk_size=7500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path):
        """This function is to load the pdf"""
        try:
            logger.info(f"Loading the pdf from {file_path}")
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Error occured during pdf loading: {e}")
            raise
    
    def split_documents(self, document):
        """This function splits the document into chunks"""
        try:
            logger.info(f"Splitting the document- {document}")
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            return splitter.split_documents(documents=document)
        except Exception as e:
            logger.error(f"Error occured during splitting the document: {e}")
            raise
