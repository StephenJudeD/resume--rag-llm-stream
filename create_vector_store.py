import os
from typing import List, Dict
from google.cloud import storage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings  # Or HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVDocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Or HuggingFaceEmbeddings

    def process_text_to_chunks(self, text: str, metadata: Dict) -> List[Document]:
        documents = []
        sections = text.split('\n## ')  # Split by markdown-style headers

        for section in sections:
            if section.strip():
                parts = section.split('\n', 1)
                section_title = parts[0].strip().replace('#', '')
                content = parts[1] if len(parts) > 1 else ""

                words = content.split()
                for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                    chunk = ' '.join(words[i:i + self.chunk_size])
                    if chunk.strip():
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "section": section_title,
                                "chunk_id": i // (self.chunk_size - self.chunk_overlap)
                            }
                        )
                        documents.append(doc)

        return documents

class VectorStoreManager:
    def __init__(self, bucket_name: str, index_path: str):
        self.bucket_name = bucket_name
        self.index_path = index_path
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def upload_to_gcs(self, local_path: str):
        try:
            for file_name in ["index.faiss", "index.pkl"]:
                local_file = f"{local_path}/{file_name}"
                blob = self.bucket.blob(f"{self.index_path}/{file_name}")
                blob.upload_from_filename(local_file)
            logger.info("Vector store uploaded successfully to GCS")
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            raise

    def download_from_gcs(self, local_path: str):
        try:
            os.makedirs(local_path, exist_ok=True)
            for file_name in ["index.faiss", "index.pkl"]:
                blob = self.bucket.blob(f"{self.index_path}/{file_name}")
                local_file = f"{local_path}/{file_name}"
                blob.download_to_filename(local_file)
            logger.info("Vector store downloaded successfully from GCS")
        except Exception as e:
            logger.error(f"Error downloading from GCS: {str(e)}")
            raise


def create_and_upload_vector_store(file_paths: List[str]):
    try:
        processor = CVDocumentProcessor()
        all_documents = []

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                documents = processor.process_text_to_chunks(text, metadata={"source": file_path})
                all_documents.extend(documents)

        vector_store = FAISS.from_documents(all_documents, processor.embeddings)  # Use processor's embeddings

        local_path = "faiss_index"
        os.makedirs(local_path, exist_ok=True)
        vector_store.save_local(local_path)

        vector_store_manager = VectorStoreManager(
            bucket_name=os.environ["GCS_BUCKET_NAME"],
            index_path=os.environ["GCS_INDEX_PATH"]
        )
        vector_store_manager.upload_to_gcs(local_path)

        return vector_store

    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise
