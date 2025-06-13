import chromadb
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from chromadb.config import Settings
from config import Config
from .llm_client import LMStudioClient

class VectorStore:
    """ChromaDB vector store for document storage and retrieval"""
    
    def __init__(self, llm_client: LMStudioClient):
        self.llm_client = llm_client
        self.db_path = Config.CHROMA_DB_PATH
        self.collection_name = Config.COLLECTION_NAME
        self.top_k = Config.TOP_K_RESULTS
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document collection"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Add documents to the vector store"""
        if not documents:
            return False
        
        try:
            print(f"Generating embeddings for {len(documents)} documents...")
            embeddings = self.llm_client.get_embeddings(documents)
            
            if not embeddings or len(embeddings) != len(documents):
                print("Failed to generate embeddings")
                return False
            
            # Generate IDs for documents
            ids = [str(uuid.uuid4()) for _ in documents]
            
            # Use provided metadata or create default
            if metadatas is None:
                metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        k = k or self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self.llm_client.get_single_embedding(query)
            
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    result = {
                        "document": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {},
                        "distance": results["distances"][0][i] if results["distances"][0] else 1.0
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "path": self.db_path
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"name": self.collection_name, "document_count": 0, "path": self.db_path}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document collection"}
            )
            print("Collection cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def load_documents_from_directory(self, directory_path: str) -> bool:
        """Load and index documents from a directory"""
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return False
        
        documents = []
        metadatas = []
        
        # Supported file extensions
        supported_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv'}
        
        try:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    if file_ext in supported_extensions:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # Split large documents into chunks
                            chunks = self._split_text(content, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
                            
                            for i, chunk in enumerate(chunks):
                                documents.append(chunk)
                                metadatas.append({
                                    "source": filename,
                                    "file_path": file_path,
                                    "chunk_index": i,
                                    "file_type": file_ext
                                })
                                
                        except Exception as e:
                            print(f"Error reading file {filename}: {e}")
                            continue
            
            if documents:
                return self.add_documents(documents, metadatas)
            else:
                print("No supported documents found in directory")
                return False
                
        except Exception as e:
            print(f"Error loading documents from directory: {e}")
            return False
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks