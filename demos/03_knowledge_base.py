#!/usr/bin/env python3
"""
Memorix SDK - Knowledge Base Demo
=================================

This demo shows how to build a searchable knowledge base using Memorix SDK:
- Document ingestion and chunking
- Semantic search across documents
- Metadata-based filtering
- Knowledge base management
- Question answering over documents

Perfect for building document search systems and Q&A applications!
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add the parent directory to the path to import memorix
sys.path.append(str(Path(__file__).parent.parent.parent))

from memorix import MemoryAPI, Config


class KnowledgeBase:
    """A searchable knowledge base built with Memorix."""
    
    def __init__(self, config: Config):
        """Initialize the knowledge base."""
        self.memory = MemoryAPI(config)
        self.documents = {}
        self.chunk_size = 500  # characters per chunk
        self.chunk_overlap = 50  # characters overlap between chunks
        
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i-1] in '.!?':
                        end = i
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_document(self, doc_id: str, content: str, metadata: Dict = None) -> List[str]:
        """Add a document to the knowledge base."""
        print(f"Processing document: {doc_id}")
        
        # Chunk the document
        chunks = self.chunk_text(content)
        print(f"  Created {len(chunks)} chunks")
        
        # Store each chunk with metadata
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "timestamp": datetime.now().isoformat()
            }
            
            # Merge with provided metadata
            if metadata:
                chunk_metadata.update(metadata)
            
            memory_id = self.memory.store(chunk, chunk_metadata)
            chunk_ids.append(memory_id)
        
        # Store document metadata
        self.documents[doc_id] = {
            "content": content,
            "chunks": chunk_ids,
            "metadata": metadata or {},
            "added_at": datetime.now().isoformat()
        }
        
        print(f"  ✅ Document {doc_id} added with {len(chunks)} chunks")
        return chunk_ids
    
    def search(self, query: str, top_k: int = 5, 
               metadata_filter: Dict = None) -> List[Dict]:
        """Search the knowledge base."""
        results = self.memory.retrieve(query, top_k=top_k * 2)  # Get more results for filtering
        
        if metadata_filter:
            # Filter results in Python
            filtered_results = []
            for result in results:
                matches = True
                for key, value in metadata_filter.items():
                    if result['metadata'].get(key) != value:
                        matches = False
                        break
                if matches:
                    filtered_results.append(result)
            return filtered_results[:top_k]
        
        return results[:top_k]
    
    def search_by_document(self, query: str, doc_id: str, top_k: int = 5) -> List[Dict]:
        """Search within a specific document."""
        results = self.memory.retrieve(query, top_k=top_k * 2)
        
        # Filter for specific document
        doc_results = [
            result for result in results 
            if result['metadata'].get('doc_id') == doc_id
        ]
        
        return doc_results[:top_k]
    
    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a specific document."""
        # Get all memories and filter by document ID
        all_memories = self.memory.list_memories(limit=1000)
        
        doc_chunks = [
            memory for memory in all_memories 
            if memory['metadata'].get('doc_id') == doc_id
        ]
        
        return doc_chunks
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks."""
        if doc_id not in self.documents:
            return False
        
        # Get all chunk IDs for this document
        chunks = self.get_document_chunks(doc_id)
        
        # Delete each chunk
        for chunk in chunks:
            self.memory.delete(chunk['memory_id'])
        
        # Remove from documents dict
        del self.documents[doc_id]
        
        print(f"✅ Document {doc_id} and {len(chunks)} chunks deleted")
        return True
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the knowledge base."""
        return [
            {
                "doc_id": doc_id,
                "metadata": doc_info["metadata"],
                "chunk_count": len(doc_info["chunks"]),
                "added_at": doc_info["added_at"]
            }
            for doc_id, doc_info in self.documents.items()
        ]
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics."""
        total_chunks = sum(len(doc["chunks"]) for doc in self.documents.values())
        total_content = sum(len(doc["content"]) for doc in self.documents.values())
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "total_content_length": total_content,
            "average_chunks_per_document": total_chunks / len(self.documents) if self.documents else 0,
            "average_content_length": total_content / len(self.documents) if self.documents else 0
        }


def setup_config():
    """Set up configuration for the knowledge base demo."""
    print("🔧 Setting up knowledge base configuration...")
    
    config = Config()
    
    # Set up vector store
    config.set('vector_store.type', 'faiss')
    config.set('vector_store.index_path', './knowledge_base_index')
    config.set('vector_store.dimension', 1536)
    
    # Set up embedder
    config.set('embedder.type', 'openai')
    config.set('embedder.model', 'text-embedding-ada-002')
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Using sentence transformers for demo...")
        config.set('embedder.type', 'sentence_transformers')
        config.set('embedder.model', 'all-MiniLM-L6-v2')
    else:
        config.set('embedder.api_key', api_key)
    
    # Set up metadata store
    config.set('metadata_store.type', 'sqlite')
    config.set('metadata_store.database_path', './knowledge_base_metadata.db')
    
    # Set up settings
    config.set('settings.max_memories', 50000)
    config.set('settings.similarity_threshold', 0.6)
    config.set('settings.default_top_k', 10)
    
    return config


def create_sample_documents() -> Dict[str, str]:
    """Create sample documents for the demo."""
    documents = {
        "python_basics": """
Python Programming Basics

Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

Key Features:
- Simple and readable syntax
- Extensive standard library
- Cross-platform compatibility
- Strong community support
- Large ecosystem of third-party packages

Basic Syntax:
Variables in Python are dynamically typed and don't require explicit declaration:
x = 10
name = "Python"
is_valid = True

Control Structures:
Python uses indentation to define code blocks:
if x > 5:
    print("x is greater than 5")
else:
    print("x is 5 or less")

Functions are defined using the 'def' keyword:
def greet(name):
    return f"Hello, {name}!"

Python is widely used in web development, data science, machine learning, automation, and many other fields.
        """,
        
        "machine_learning": """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed for every task.

Types of Machine Learning:
1. Supervised Learning: Learning from labeled training data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through interaction with an environment

Common Algorithms:
- Linear Regression: For predicting continuous values
- Logistic Regression: For binary classification
- Decision Trees: For both classification and regression
- Random Forests: Ensemble method using multiple decision trees
- Support Vector Machines: For classification with clear margins
- Neural Networks: Deep learning models for complex patterns

Python Libraries for ML:
- scikit-learn: Traditional machine learning algorithms
- TensorFlow: Deep learning framework by Google
- PyTorch: Deep learning framework by Facebook
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Data visualization

The machine learning workflow typically involves:
1. Data collection and preprocessing
2. Feature engineering
3. Model selection and training
4. Model evaluation and validation
5. Deployment and monitoring
        """,
        
        "data_structures": """
Data Structures in Computer Science

Data structures are ways of organizing and storing data for efficient access and modification. They are fundamental to computer science and programming.

Basic Data Structures:
1. Arrays: Contiguous memory locations storing elements
2. Linked Lists: Elements connected by pointers
3. Stacks: Last-in-first-out (LIFO) structure
4. Queues: First-in-first-out (FIFO) structure
5. Trees: Hierarchical structure with nodes
6. Graphs: Collection of nodes connected by edges
7. Hash Tables: Key-value pairs with fast lookup

Advanced Data Structures:
- Binary Search Trees: Ordered tree structure
- Heaps: Specialized tree-based data structure
- Tries: Tree-like structure for string operations
- B-Trees: Self-balancing tree structures
- Red-Black Trees: Self-balancing binary search trees

Time Complexity:
Understanding time complexity is crucial:
- O(1): Constant time operations
- O(log n): Logarithmic time operations
- O(n): Linear time operations
- O(n log n): Linearithmic time operations
- O(n²): Quadratic time operations

Choosing the right data structure depends on:
- The type of operations you need to perform
- The frequency of different operations
- Memory constraints
- Performance requirements
        """,
        
        "web_development": """
Modern Web Development

Web development involves creating websites and web applications that run on the internet. Modern web development encompasses both frontend and backend development.

Frontend Development:
Frontend development focuses on the user interface and user experience:
- HTML: Structure and content
- CSS: Styling and layout
- JavaScript: Interactivity and dynamic content
- React: Popular JavaScript library for building user interfaces
- Vue.js: Progressive JavaScript framework
- Angular: Full-featured framework by Google

Backend Development:
Backend development handles server-side logic and data management:
- Python: Django, Flask, FastAPI
- Node.js: Express.js, NestJS
- Java: Spring Boot, Jakarta EE
- PHP: Laravel, Symfony
- Ruby: Ruby on Rails
- Go: Gin, Echo

Database Technologies:
- SQL Databases: PostgreSQL, MySQL, SQLite
- NoSQL Databases: MongoDB, Redis, Cassandra
- Graph Databases: Neo4j, ArangoDB
- Time Series Databases: InfluxDB, TimescaleDB

Web Development Tools:
- Version Control: Git, GitHub, GitLab
- Package Managers: npm, yarn, pip, composer
- Build Tools: Webpack, Vite, Parcel
- Testing: Jest, PyTest, Selenium
- Deployment: Docker, Kubernetes, AWS, Heroku

Modern web development emphasizes:
- Responsive design for mobile devices
- Progressive Web Apps (PWAs)
- API-first development
- Microservices architecture
- DevOps and CI/CD practices
        """
    }
    
    return documents


def demo_document_ingestion(kb: KnowledgeBase):
    """Demonstrate document ingestion."""
    print("\n📚 Demo: Document Ingestion")
    print("=" * 50)
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Add documents to knowledge base
    for doc_id, content in documents.items():
        metadata = {
            "category": "programming",
            "difficulty": "intermediate",
            "tags": ["tutorial", "reference"],
            "author": "demo_author"
        }
        
        # Add category-specific metadata
        if "python" in doc_id:
            metadata["language"] = "python"
            metadata["difficulty"] = "beginner"
        elif "machine_learning" in doc_id:
            metadata["category"] = "ai"
            metadata["difficulty"] = "advanced"
        elif "data_structures" in doc_id:
            metadata["category"] = "computer_science"
            metadata["difficulty"] = "intermediate"
        elif "web_development" in doc_id:
            metadata["category"] = "web"
            metadata["difficulty"] = "intermediate"
        
        kb.add_document(doc_id, content, metadata)
    
    print(f"\n✅ Added {len(documents)} documents to knowledge base")


def demo_semantic_search(kb: KnowledgeBase):
    """Demonstrate semantic search capabilities."""
    print("\n🔍 Demo: Semantic Search")
    print("=" * 50)
    
    # Test various search queries
    search_queries = [
        "How to learn Python programming?",
        "What are the best machine learning algorithms?",
        "Explain data structures and algorithms",
        "Modern web development frameworks",
        "Time complexity of algorithms",
        "Deep learning vs traditional machine learning"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = kb.search(query, top_k=3)
        
        if results:
            print(f"Found {len(results)} relevant results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Document: {result['metadata'].get('doc_id', 'Unknown')}")
                print(f"     Chunk: {result['metadata'].get('chunk_index', 'N/A')}")
                print(f"     Content: {result['content'][:100]}...")
                print(f"     Similarity: {result['similarity']:.3f}")
                print()
        else:
            print("  No relevant results found.")


def demo_metadata_filtering(kb: KnowledgeBase):
    """Demonstrate metadata-based filtering."""
    print("\n🏷️  Demo: Metadata Filtering")
    print("=" * 50)
    
    # Search with different metadata filters
    filters = [
        {"difficulty": "beginner"},
        {"category": "ai"},
        {"language": "python"},
        {"category": "web"}
    ]
    
    query = "programming concepts"
    
    for filter_dict in filters:
        filter_name = ", ".join(f"{k}={v}" for k, v in filter_dict.items())
        print(f"\nSearching for '{query}' with filter: {filter_name}")
        
        results = kb.search(query, top_k=3, metadata_filter=filter_dict)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['metadata'].get('doc_id', 'Unknown')}")
                print(f"     {result['content'][:60]}...")
        else:
            print("  No results found with this filter.")


def demo_document_specific_search(kb: KnowledgeBase):
    """Demonstrate searching within specific documents."""
    print("\n📄 Demo: Document-Specific Search")
    print("=" * 50)
    
    # Search within specific documents
    document_searches = [
        ("python_basics", "syntax and variables"),
        ("machine_learning", "neural networks"),
        ("data_structures", "time complexity"),
        ("web_development", "frontend frameworks")
    ]
    
    for doc_id, query in document_searches:
        print(f"\nSearching in '{doc_id}' for: '{query}'")
        results = kb.search_by_document(query, doc_id, top_k=3)
        
        if results:
            print(f"Found {len(results)} results in {doc_id}:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Chunk {result['metadata'].get('chunk_index', 'N/A')}")
                print(f"     {result['content'][:80]}...")
                print(f"     Similarity: {result['similarity']:.3f}")
        else:
            print(f"  No results found in {doc_id}.")


def demo_knowledge_base_management(kb: KnowledgeBase):
    """Demonstrate knowledge base management features."""
    print("\n⚙️  Demo: Knowledge Base Management")
    print("=" * 50)
    
    # List all documents
    print("Documents in knowledge base:")
    documents = kb.list_documents()
    for doc in documents:
        print(f"  📄 {doc['doc_id']}")
        print(f"     Chunks: {doc['chunk_count']}")
        print(f"     Category: {doc['metadata'].get('category', 'N/A')}")
        print(f"     Difficulty: {doc['metadata'].get('difficulty', 'N/A')}")
        print(f"     Added: {doc['added_at']}")
        print()
    
    # Get statistics
    stats = kb.get_statistics()
    print("Knowledge Base Statistics:")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Total Content Length: {stats['total_content_length']:,} characters")
    print(f"  Average Chunks per Document: {stats['average_chunks_per_document']:.1f}")
    print(f"  Average Content Length: {stats['average_content_length']:.0f} characters")


def demo_question_answering(kb: KnowledgeBase):
    """Demonstrate question answering capabilities."""
    print("\n❓ Demo: Question Answering")
    print("=" * 50)
    
    questions = [
        "What is Python and what are its key features?",
        "How does machine learning work?",
        "What are the different types of data structures?",
        "What tools are used in modern web development?",
        "What is the difference between supervised and unsupervised learning?",
        "How do you implement a binary search tree?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        
        # Search for relevant content
        results = kb.search(question, top_k=3)
        
        if results:
            print("A: Based on the knowledge base:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['content'][:120]}...")
                print(f"     Source: {result['metadata'].get('doc_id', 'Unknown')}")
        else:
            print("A: No relevant information found in the knowledge base.")


def demo_interactive_search(kb: KnowledgeBase):
    """Demonstrate interactive search functionality."""
    print("\n🎮 Demo: Interactive Search")
    print("=" * 50)
    print("You can now search the knowledge base! Type 'quit' to exit.")
    print("Available commands:")
    print("  - Type any search query")
    print("  - 'list' to see all documents")
    print("  - 'stats' to see statistics")
    print("  - 'filter:category=ai' to filter by metadata")
    
    while True:
        try:
            query = input("\nSearch: ").strip()
            
            if query.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            if query.lower() == 'list':
                documents = kb.list_documents()
                print(f"\nDocuments ({len(documents)}):")
                for doc in documents:
                    print(f"  - {doc['doc_id']} ({doc['chunk_count']} chunks)")
                continue
            
            if query.lower() == 'stats':
                stats = kb.get_statistics()
                print(f"\nStatistics:")
                print(f"  Documents: {stats['total_documents']}")
                print(f"  Chunks: {stats['total_chunks']}")
                print(f"  Content: {stats['total_content_length']:,} characters")
                continue
            
            # Check for metadata filter
            metadata_filter = None
            if query.startswith('filter:'):
                filter_part = query[7:]  # Remove 'filter:' prefix
                search_part = ""
                
                if ' ' in filter_part:
                    filter_part, search_part = filter_part.split(' ', 1)
                
                # Parse filter (simple key=value format)
                if '=' in filter_part:
                    key, value = filter_part.split('=', 1)
                    metadata_filter = {key.strip(): value.strip()}
                    query = search_part
                else:
                    print("Invalid filter format. Use 'filter:key=value'")
                    continue
            
            # Perform search
            results = kb.search(query, top_k=5, metadata_filter=metadata_filter)
            
            if results:
                print(f"\nFound {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Document: {result['metadata'].get('doc_id', 'Unknown')}")
                    print(f"   Chunk: {result['metadata'].get('chunk_index', 'N/A')}")
                    print(f"   Similarity: {result['similarity']:.3f}")
                    print(f"   Content: {result['content']}")
            else:
                print("No results found.")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def cleanup():
    """Clean up demo files."""
    print("\n🧹 Cleaning up demo files...")
    
    files_to_remove = [
        './knowledge_base_index',
        './knowledge_base_metadata.db'
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            print(f"  Removed: {file_path}")


def main():
    """Main demo function."""
    print("📚 Memorix SDK - Knowledge Base Demo")
    print("=" * 50)
    print("This demo shows how to build a searchable knowledge base")
    print("using Memorix SDK!")
    
    try:
        # Set up configuration
        config = setup_config()
        
        # Initialize knowledge base
        print("\n🚀 Initializing Knowledge Base...")
        kb = KnowledgeBase(config)
        print("✅ Knowledge base initialized successfully!")
        
        # Run demos
        demo_document_ingestion(kb)
        demo_semantic_search(kb)
        demo_metadata_filtering(kb)
        demo_document_specific_search(kb)
        demo_knowledge_base_management(kb)
        demo_question_answering(kb)
        
        # Interactive demo
        print("\n" + "="*50)
        print("🎮 INTERACTIVE DEMO")
        print("="*50)
        demo_interactive_search(kb)
        
        print("\n🎉 Knowledge base demo completed successfully!")
        print("\nWhat you learned:")
        print("✅ How to build a searchable knowledge base")
        print("✅ How to ingest and chunk documents")
        print("✅ How to perform semantic search")
        print("✅ How to use metadata filtering")
        print("✅ How to manage knowledge base content")
        print("✅ How to implement question answering")
        
        print("\n🚀 Ready to build your own knowledge base!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check your configuration and API keys.")
        return 1
    
    finally:
        # Ask user if they want to clean up
        response = input("\n🧹 Clean up demo files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup()
            print("✅ Cleanup completed!")
        else:
            print("💡 Demo files preserved for inspection.")
    
    return 0


if __name__ == "__main__":
    exit(main()) 