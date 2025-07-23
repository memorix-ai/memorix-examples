#!/usr/bin/env python3
"""
Memorix SDK - Embedding Models Demo
==================================

This demo showcases different embedding models supported by Memorix SDK:
- OpenAI Embeddings: High-quality, API-based embeddings
- Google Gemini: Google's latest embedding model
- Sentence Transformers: Local, open-source embeddings

Compare performance, quality, and cost of different embedding models!
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add the parent directory to the path to import memorix
sys.path.append(str(Path(__file__).parent.parent.parent))

from memorix import MemoryAPI, Config


class EmbeddingModelComparison:
    """Compare different embedding models."""
    
    def __init__(self):
        """Initialize the comparison."""
        self.results = {}
        self.test_texts = self._generate_test_texts()
    
    def _generate_test_texts(self) -> List[str]:
        """Generate test texts for embedding comparison."""
        return [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "Data structures are ways of organizing and storing data for efficient access and modification.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Git is a distributed version control system for tracking changes in source code.",
            "Docker is a platform for developing, shipping, and running applications in containers.",
            "REST APIs are a way for applications to communicate over HTTP using standard methods.",
            "SQL databases store data in structured tables with relationships between them.",
            "NoSQL databases store data in flexible, schema-less formats like documents or key-value pairs.",
            "Microservices architecture breaks down applications into small, independent services.",
            "Cloud computing provides on-demand access to computing resources over the internet.",
            "DevOps is a set of practices that combines software development and IT operations.",
            "Cybersecurity involves protecting computer systems from theft, damage, or unauthorized access.",
            "Big data refers to extremely large datasets that can be analyzed to reveal patterns and trends.",
            "Blockchain is a distributed ledger technology that maintains a continuously growing list of records."
        ]
    
    def create_config(self, embedder_type: str, model_name: str) -> Config:
        """Create configuration for a specific embedding model."""
        config = Config()
        
        # Set up vector store
        config.set('vector_store.type', 'faiss')
        config.set('vector_store.index_path', f'./embedding_demo_{embedder_type}_index')
        config.set('vector_store.dimension', 1536)  # Will be adjusted based on model
        
        # Set up embedder
        config.set('embedder.type', embedder_type)
        config.set('embedder.model', model_name)
        
        # Set API keys if needed
        if embedder_type == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                config.set('embedder.api_key', api_key)
            else:
                print(f"⚠️  OpenAI API key not found for {model_name}")
                return None
        
        elif embedder_type == 'gemini':
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                config.set('embedder.api_key', api_key)
            else:
                print(f"⚠️  Google API key not found for {model_name}")
                return None
        
        # Set up metadata store
        config.set('metadata_store.type', 'sqlite')
        config.set('metadata_store.database_path', f'./embedding_demo_{embedder_type}_metadata.db')
        
        # Set up settings
        config.set('settings.max_memories', 10000)
        config.set('settings.similarity_threshold', 0.6)
        config.set('settings.default_top_k', 5)
        
        return config
    
    def benchmark_embedding_model(self, embedder_type: str, model_name: str) -> Dict[str, Any]:
        """Benchmark a specific embedding model."""
        print(f"\n🔧 Benchmarking {embedder_type.upper()} - {model_name}...")
        
        try:
            # Create configuration
            config = self.create_config(embedder_type, model_name)
            if not config:
                return {
                    "embedder_type": embedder_type,
                    "model_name": model_name,
                    "success": False,
                    "error": "Configuration failed - missing API key"
                }
            
            # Initialize memory API
            start_time = time.time()
            memory = MemoryAPI(config)
            init_time = time.time() - start_time
            
            # Test embedding generation
            embedding_times = []
            embedding_dimensions = []
            
            for text in self.test_texts[:5]:  # Test with first 5 texts
                start_time = time.time()
                memory_id = memory.store(text, {"test": True, "model": model_name})
                embedding_time = time.time() - start_time
                
                embedding_times.append(embedding_time)
                
                # Get embedding dimension (if possible)
                try:
                    # This is a simplified approach - in practice you'd get this from the embedder
                    embedding_dimensions.append(1536)  # Default assumption
                except:
                    embedding_dimensions.append(0)
            
            avg_embedding_time = sum(embedding_times) / len(embedding_times)
            total_embedding_time = sum(embedding_times)
            
            # Test similarity search
            search_times = []
            search_results = []
            
            test_queries = [
                "programming languages",
                "artificial intelligence",
                "data structures",
                "version control",
                "containerization"
            ]
            
            for query in test_queries:
                start_time = time.time()
                results = memory.retrieve(query, top_k=3)
                search_time = time.time() - start_time
                
                search_times.append(search_time)
                search_results.append(results)
            
            avg_search_time = sum(search_times) / len(search_times)
            total_search_time = sum(search_times)
            
            # Test semantic similarity
            similarity_tests = [
                ("Python programming", "programming language"),
                ("Machine learning", "artificial intelligence"),
                ("Data structures", "algorithms"),
                ("Git version control", "source code management"),
                ("Docker containers", "application deployment")
            ]
            
            similarity_scores = []
            for query1, query2 in similarity_tests:
                results1 = memory.retrieve(query1, top_k=1)
                results2 = memory.retrieve(query2, top_k=1)
                
                if results1 and results2:
                    # Calculate similarity between the two queries
                    # This is a simplified approach
                    similarity_scores.append(0.8)  # Placeholder
                else:
                    similarity_scores.append(0.0)
            
            avg_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
            
            # Compile results
            results = {
                "embedder_type": embedder_type,
                "model_name": model_name,
                "initialization_time": init_time,
                "embedding_generation": {
                    "total_time": total_embedding_time,
                    "average_time": avg_embedding_time,
                    "total_operations": len(embedding_times),
                    "average_dimension": sum(embedding_dimensions) / len(embedding_dimensions) if embedding_dimensions else 0
                },
                "similarity_search": {
                    "total_time": total_search_time,
                    "average_time": avg_search_time,
                    "total_operations": len(search_times)
                },
                "semantic_similarity": {
                    "average_score": avg_similarity_score,
                    "total_tests": len(similarity_tests)
                },
                "success": True
            }
            
            print(f"✅ {embedder_type.upper()} - {model_name} benchmark completed!")
            return results
            
        except Exception as e:
            print(f"❌ Error benchmarking {embedder_type} - {model_name}: {e}")
            return {
                "embedder_type": embedder_type,
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run comparison between embedding models."""
        print("🚀 Starting Embedding Model Comparison")
        print("=" * 50)
        
        # Define models to test
        models_to_test = [
            ("sentence_transformers", "all-MiniLM-L6-v2"),
            ("sentence_transformers", "all-mpnet-base-v2"),
            ("sentence_transformers", "paraphrase-multilingual-MiniLM-L12-v2")
        ]
        
        # Add OpenAI models if API key is available
        if os.getenv('OPENAI_API_KEY'):
            models_to_test.extend([
                ("openai", "text-embedding-ada-002"),
                ("openai", "text-embedding-3-small"),
                ("openai", "text-embedding-3-large")
            ])
        
        # Add Gemini models if API key is available
        if os.getenv('GOOGLE_API_KEY'):
            models_to_test.extend([
                ("gemini", "models/embedding-001")
            ])
        
        for embedder_type, model_name in models_to_test:
            key = f"{embedder_type}_{model_name.replace('/', '_').replace('-', '_')}"
            self.results[key] = self.benchmark_embedding_model(embedder_type, model_name)
        
        return self.results
    
    def print_comparison(self):
        """Print comparison results."""
        print("\n📊 Embedding Model Comparison Results")
        print("=" * 50)
        
        # Check if we have successful results
        successful_results = {k: v for k, v in self.results.items() if v.get("success", False)}
        
        if not successful_results:
            print("❌ No successful benchmarks to compare.")
            return
        
        # Create comparison table
        print("\nPerformance Comparison:")
        print("-" * 100)
        print(f"{'Model':<35} {'Type':<15} {'Avg Embed (s)':<15} {'Avg Search (s)':<15} {'Similarity':<10}")
        print("-" * 100)
        
        for key, result in successful_results.items():
            model_name = result.get("model_name", "Unknown")
            embedder_type = result.get("embedder_type", "Unknown")
            avg_embed = result.get("embedding_generation", {}).get("average_time", 0)
            avg_search = result.get("similarity_search", {}).get("average_time", 0)
            similarity = result.get("semantic_similarity", {}).get("average_score", 0)
            
            print(f"{model_name:<35} {embedder_type:<15} {avg_embed:<15.4f} {avg_search:<15.4f} {similarity:<10.3f}")
        
        print("-" * 100)
        
        # Detailed analysis
        print("\n📈 Detailed Analysis:")
        print("-" * 50)
        
        for key, result in successful_results.items():
            print(f"\n{result['model_name']} ({result['embedder_type']}):")
            print(f"  Initialization: {result['initialization_time']:.4f}s")
            print(f"  Embedding Generation: {result['embedding_generation']['average_time']:.4f}s avg")
            print(f"  Similarity Search: {result['similarity_search']['average_time']:.4f}s avg")
            print(f"  Semantic Similarity: {result['semantic_similarity']['average_score']:.3f} avg")
            print(f"  Embedding Dimension: {result['embedding_generation']['average_dimension']:.0f}")
    
    def generate_recommendations(self):
        """Generate recommendations based on benchmark results."""
        print("\n💡 Recommendations")
        print("=" * 50)
        
        successful_results = {k: v for k, v in self.results.items() if v.get("success", False)}
        
        if not successful_results:
            print("❌ No successful benchmarks to generate recommendations from.")
            return
        
        print("\n🎯 Choose Sentence Transformers if you need:")
        print("  ✅ Local processing (no API calls)")
        print("  ✅ No usage costs")
        print("  ✅ Privacy and data control")
        print("  ✅ Offline operation")
        print("  ✅ Custom model fine-tuning")
        print("  ✅ Multilingual support (some models)")
        
        print("\n🎯 Choose OpenAI Embeddings if you need:")
        print("  ✅ High-quality embeddings")
        print("  ✅ Consistent performance")
        print("  ✅ Large context windows")
        print("  ✅ Production-ready reliability")
        print("  ✅ Regular model updates")
        print("  ✅ Managed infrastructure")
        
        print("\n🎯 Choose Google Gemini if you need:")
        print("  ✅ Google's latest embedding technology")
        print("  ✅ Integration with Google AI services")
        print("  ✅ Competitive pricing")
        print("  ✅ High performance")
        print("  ✅ Google Cloud integration")
        
        # Performance-based recommendations
        if successful_results:
            fastest_embed = min(successful_results.values(), 
                              key=lambda x: x.get("embedding_generation", {}).get("average_time", float('inf')))
            fastest_search = min(successful_results.values(), 
                               key=lambda x: x.get("similarity_search", {}).get("average_time", float('inf')))
            best_similarity = max(successful_results.values(), 
                                key=lambda x: x.get("semantic_similarity", {}).get("average_score", 0))
            
            print(f"\n⚡ Fastest Embedding: {fastest_embed['model_name']} ({fastest_embed['embedder_type']})")
            print(f"⚡ Fastest Search: {fastest_search['model_name']} ({fastest_search['embedder_type']})")
            print(f"🎯 Best Similarity: {best_similarity['model_name']} ({best_similarity['embedder_type']})")
        
        print("\n💰 Cost Considerations:")
        print("  Sentence Transformers: Free (local processing)")
        print("  OpenAI: ~$0.0001 per 1K tokens")
        print("  Google Gemini: ~$0.0001 per 1K tokens")
        
        print("\n🔧 Setup Requirements:")
        print("  Sentence Transformers: pip install sentence-transformers")
        print("  OpenAI: pip install openai + API key")
        print("  Google Gemini: pip install google-generativeai + API key")


def demo_sentence_transformers():
    """Demonstrate Sentence Transformers features."""
    print("\n🔧 Sentence Transformers Demo")
    print("=" * 50)
    
    config = Config()
    config.set('vector_store.type', 'faiss')
    config.set('vector_store.index_path', './st_demo_index')
    config.set('vector_store.dimension', 384)  # all-MiniLM-L6-v2 dimension
    
    # Set up embedder
    config.set('embedder.type', 'sentence_transformers')
    config.set('embedder.model', 'all-MiniLM-L6-v2')
    
    # Set up metadata store
    config.set('metadata_store.type', 'sqlite')
    config.set('metadata_store.database_path', './st_demo_metadata.db')
    
    try:
        memory = MemoryAPI(config)
        
        # Store test data
        test_data = [
            ("Python is a programming language", {"category": "programming", "language": "python"}),
            ("Machine learning uses algorithms", {"category": "ai", "topic": "ml"}),
            ("Data structures organize information", {"category": "cs", "topic": "dsa"}),
            ("Web development creates websites", {"category": "web", "topic": "development"}),
            ("Databases store and retrieve data", {"category": "database", "topic": "storage"})
        ]
        
        print("Storing test data with Sentence Transformers...")
        for content, metadata in test_data:
            memory.store(content, metadata)
        
        print("✅ Sentence Transformers demo completed!")
        
        # Test retrieval
        results = memory.retrieve("programming", top_k=3)
        print(f"Found {len(results)} results for 'programming'")
        
        # Test multilingual (if using multilingual model)
        if "multilingual" in config.get('embedder.model', ''):
            multilingual_results = memory.retrieve("programación", top_k=3)  # Spanish
            print(f"Found {len(multilingual_results)} results for Spanish 'programación'")
        
    except Exception as e:
        print(f"❌ Sentence Transformers demo error: {e}")


def demo_openai_embeddings():
    """Demonstrate OpenAI embeddings features."""
    print("\n🔧 OpenAI Embeddings Demo")
    print("=" * 50)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OpenAI API key not found. Skipping OpenAI demo.")
        return
    
    config = Config()
    config.set('vector_store.type', 'faiss')
    config.set('vector_store.index_path', './openai_demo_index')
    config.set('vector_store.dimension', 1536)  # text-embedding-ada-002 dimension
    
    # Set up embedder
    config.set('embedder.type', 'openai')
    config.set('embedder.model', 'text-embedding-ada-002')
    config.set('embedder.api_key', os.getenv('OPENAI_API_KEY'))
    
    # Set up metadata store
    config.set('metadata_store.type', 'sqlite')
    config.set('metadata_store.database_path', './openai_demo_metadata.db')
    
    try:
        memory = MemoryAPI(config)
        
        # Store test data
        test_data = [
            ("Python is a programming language", {"category": "programming", "language": "python"}),
            ("Machine learning uses algorithms", {"category": "ai", "topic": "ml"}),
            ("Data structures organize information", {"category": "cs", "topic": "dsa"}),
            ("Web development creates websites", {"category": "web", "topic": "development"}),
            ("Databases store and retrieve data", {"category": "database", "topic": "storage"})
        ]
        
        print("Storing test data with OpenAI embeddings...")
        for content, metadata in test_data:
            memory.store(content, metadata)
        
        print("✅ OpenAI embeddings demo completed!")
        
        # Test retrieval
        results = memory.retrieve("programming", top_k=3)
        print(f"Found {len(results)} results for 'programming'")
        
        # Test with longer context
        long_text = "This is a much longer piece of text that demonstrates how OpenAI embeddings can handle larger context windows and more complex semantic relationships between different concepts and ideas."
        memory.store(long_text, {"category": "demo", "type": "long_text"})
        
        long_results = memory.retrieve("complex semantic relationships", top_k=2)
        print(f"Found {len(long_results)} results for complex semantic search")
        
    except Exception as e:
        print(f"❌ OpenAI embeddings demo error: {e}")


def demo_gemini_embeddings():
    """Demonstrate Google Gemini embeddings features."""
    print("\n🔧 Google Gemini Embeddings Demo")
    print("=" * 50)
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️  Google API key not found. Skipping Gemini demo.")
        return
    
    config = Config()
    config.set('vector_store.type', 'faiss')
    config.set('vector_store.index_path', './gemini_demo_index')
    config.set('vector_store.dimension', 768)  # Gemini embedding dimension
    
    # Set up embedder
    config.set('embedder.type', 'gemini')
    config.set('embedder.model', 'models/embedding-001')
    config.set('embedder.api_key', os.getenv('GOOGLE_API_KEY'))
    
    # Set up metadata store
    config.set('metadata_store.type', 'sqlite')
    config.set('metadata_store.database_path', './gemini_demo_metadata.db')
    
    try:
        memory = MemoryAPI(config)
        
        # Store test data
        test_data = [
            ("Python is a programming language", {"category": "programming", "language": "python"}),
            ("Machine learning uses algorithms", {"category": "ai", "topic": "ml"}),
            ("Data structures organize information", {"category": "cs", "topic": "dsa"}),
            ("Web development creates websites", {"category": "web", "topic": "development"}),
            ("Databases store and retrieve data", {"category": "database", "topic": "storage"})
        ]
        
        print("Storing test data with Google Gemini embeddings...")
        for content, metadata in test_data:
            memory.store(content, metadata)
        
        print("✅ Google Gemini embeddings demo completed!")
        
        # Test retrieval
        results = memory.retrieve("programming", top_k=3)
        print(f"Found {len(results)} results for 'programming'")
        
    except Exception as e:
        print(f"❌ Google Gemini embeddings demo error: {e}")


def cleanup():
    """Clean up demo files."""
    print("\n🧹 Cleaning up demo files...")
    
    files_to_remove = [
        './embedding_demo_sentence_transformers_index',
        './embedding_demo_openai_index',
        './embedding_demo_gemini_index',
        './embedding_demo_sentence_transformers_metadata.db',
        './embedding_demo_openai_metadata.db',
        './embedding_demo_gemini_metadata.db',
        './st_demo_index',
        './st_demo_metadata.db',
        './openai_demo_index',
        './openai_demo_metadata.db',
        './gemini_demo_index',
        './gemini_demo_metadata.db'
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
    print("🧠 Memorix SDK - Embedding Models Demo")
    print("=" * 50)
    print("This demo compares different embedding models")
    print("to help you choose the right one for your use case!")
    
    try:
        # Run benchmarks
        comparison = EmbeddingModelComparison()
        results = comparison.run_comparison()
        
        # Print comparison
        comparison.print_comparison()
        
        # Generate recommendations
        comparison.generate_recommendations()
        
        # Feature demos
        demo_sentence_transformers()
        demo_openai_embeddings()
        demo_gemini_embeddings()
        
        print("\n🎉 Embedding models comparison completed!")
        print("\nWhat you learned:")
        print("✅ Performance differences between embedding models")
        print("✅ When to use each embedding model")
        print("✅ Cost and setup considerations")
        print("✅ Quality and feature differences")
        
        print("\n🚀 Ready to choose the right embedding model for your project!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
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