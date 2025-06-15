import chromadb
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
from chromadb.config import Settings
from config import Config
from .llm_client import LMStudioClient

# PDF processing imports
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF processing libraries not available. Install PyPDF2 and PyMuPDF for PDF support.")

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
    
    def _extract_text_from_pdf_pypdf2(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"PyPDF2 extraction failed for {file_path}: {e}")
            return ""
    
    def _extract_text_from_pdf_pymupdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF (more robust)"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"PyMuPDF extraction failed for {file_path}: {e}")
            return ""
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using available libraries"""
        if not PDF_AVAILABLE:
            print(f"PDF libraries not available. Cannot process {file_path}")
            return ""
        
        # Try PyMuPDF first (more robust), then fallback to PyPDF2
        text = self._extract_text_from_pdf_pymupdf(file_path)
        if not text:
            text = self._extract_text_from_pdf_pypdf2(file_path)
        
        if not text:
            print(f"Failed to extract text from PDF: {file_path}")
        
        return text
    
    def load_documents_from_directory(self, directory_path: str) -> bool:
        """Load and index documents from a directory"""
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return False
        
        documents = []
        metadatas = []
        
        # Supported file extensions (now including PDF)
        supported_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.pdf'}
        
        try:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                
                if os.path.isfile(file_path):
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    if file_ext in supported_extensions:
                        print(f"Processing file: {filename}")
                        
                        try:
                            # Handle PDF files differently
                            if file_ext == '.pdf':
                                if not PDF_AVAILABLE:
                                    print(f"Skipping PDF {filename} - PDF libraries not available")
                                    continue
                                content = self._extract_text_from_pdf(file_path)
                            else:
                                # Handle text-based files
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                            
                            if not content.strip():
                                print(f"No content extracted from {filename}")
                                continue
                            
                            # Split large documents into chunks
                            chunks = self._split_text(content, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
                            
                            for i, chunk in enumerate(chunks):
                                if chunk.strip():  # Only add non-empty chunks
                                    documents.append(chunk)
                                    metadatas.append({
                                        "source": filename,
                                        "file_path": file_path,
                                        "chunk_index": i,
                                        "file_type": file_ext,
                                        "total_chunks": len(chunks)
                                    })
                            
                            print(f"✅ Processed {filename}: {len(chunks)} chunks")
                                
                        except Exception as e:
                            print(f"❌ Error reading file {filename}: {e}")
                            continue
                    else:
                        print(f"Skipping unsupported file: {filename}")
            
            if documents:
                print(f"\n📚 Total chunks to index: {len(documents)}")
                return self.add_documents(documents, metadatas)
            else:
                print("❌ No supported documents found in directory")
                return False
                
        except Exception as e:
            print(f"Error loading documents from directory: {e}")
            return False
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks with improved logic"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within a reasonable range
                best_break = end
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        # Check if it's likely end of sentence (followed by space or newline)
                        if i + 1 < len(text) and text[i + 1] in ' \n\t':
                            best_break = i + 1
                            break
                    elif text[i] in '\n':
                        # Break at paragraph boundary
                        best_break = i + 1
                        break
                end = best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            if end >= len(text):
                break
            start = max(start + 1, end - chunk_overlap)
        
        return chunks
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        try:
            results = self.collection.get(
                include=["metadatas"]
            )
            
            if not results["metadatas"]:
                return {"total_chunks": 0, "files": {}, "file_types": {}}
            
            file_stats = {}
            type_stats = {}
            
            for metadata in results["metadatas"]:
                filename = metadata.get("source", "unknown")
                file_type = metadata.get("file_type", "unknown")
                
                # Count by file
                if filename not in file_stats:
                    file_stats[filename] = 0
                file_stats[filename] += 1
                
                # Count by type
                if file_type not in type_stats:
                    type_stats[file_type] = 0
                type_stats[file_type] += 1
            
            return {
                "total_chunks": len(results["metadatas"]),
                "files": file_stats,
                "file_types": type_stats
            }
            
        except Exception as e:
            print(f"Error getting document statistics: {e}")
            return {"total_chunks": 0, "files": {}, "file_types": {}}