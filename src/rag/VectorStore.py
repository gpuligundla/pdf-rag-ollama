"""This module is responsible for creating vector embeddings"""

import logging
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = os.path.join("data", "vectors")

class VectorStore:
    """Manages vector embeddings in the ChromaDB"""
    def __init__(self, embedding_model="nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model = embedding_model)
        self.vectordb = None
    
    def create_vector_db(self, documents, collection_name="pdf-rag"):
        """This function creates the vector store"""
        try:
            logger.info("creating the vector database")
            self.vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=PERSIST_DIRECTORY,
            )

            return self.vectordb
        except Exception as e:
            logger.error(f"Error occurred during vector store creation: {e}")
            raise
    
    def delete_collection(self):
        """This function deletes the vector store"""
        if self.vectordb:
            try:
                logger.info(f"Deleting the vector databse")
                self.vectordb.delete_collection()
                self.vectordb = None
            except Exception as e:
                logger.error(f"Error occurred during deleting the vector collection: {e}")
                raise
