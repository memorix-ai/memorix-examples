#!/usr/bin/env python3
"""
Memorix SDK - Vector Store Comparison Demo
==========================================

This demo compares different vector stores supported by Memorix SDK:
- FAISS: Fast similarity search
- Qdrant: Vector database with advanced features
- Performance benchmarking
- Feature comparison

Perfect for choosing the right vector store for your use case!
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add the parent directory to the path to import memorix
sys.path.append(str(Path(__file__).parent.parent.parent))

from memorix import MemoryAPI, Config


class VectorStoreBenchmark:
    """Benchmark different vector stores."""
    
    def __init__(self):
        """Initialize the benchmark."""
        self.results = {}
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> List[Dict[str, Any]]:
        """Generate test data for benchmarking."""
        return [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"category": "programming", "language": "python", "difficulty": "beginner"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "metadata": {"category": "ai", "topic": "machine_learning", "difficulty": "intermediate"}
            },
            {
                "content": "Data structures are ways of organizing and storing data for efficient access and modification.",
                "metadata": {"category": "computer_science", "topic": "data_structures", "difficulty": "intermediate"}
            },
            {
                "content": "Neural networks are computing systems inspired by biological neural networks.",
                "metadata": {"category": "ai", "topic": "neural_networks", "difficulty": "advanced"}
            },
            {
                "content": "Git is a distributed version control system for tracking changes in source code.",
                "metadata": {"category": "tools", "topic": "version_control", "difficulty": "beginner"}
            },
            {
                "content": "Docker is a platform for developing, shipping, and running applications in containers.",
                "metadata": {"category": "tools", "topic": "containerization", "difficulty": "intermediate"}
            },
            {
                "content": "REST APIs are a way for applications to communicate over HTTP using standard methods.",
                "metadata": {"category": "web", "topic": "apis", "difficulty": "intermediate"}
            },
            {
                "content": "SQL databases store data in structured tables with relationships between them.",
                "metadata": {"category": "database", "topic": "sql", "difficulty": "intermediate"}
            },
            {
                "content": "NoSQL databases store data in flexible, schema-less formats like documents or key-value pairs.",
                "metadata": {"category": "database", "topic": "nosql", "difficulty": "intermediate"}
            },
            {
                "content": "Microservices architecture breaks down applications into small, independent services.",
                "metadata": {"category": "architecture", "topic": "microservices", "difficulty": "advanced"}
            }
        ]
    
    def create_config(self, vector_store_type: str) -> Config:
        """Create configuration for a specific vector store."""
        # Load base configuration from YAML file
        config_path = Path(__file__).parent.parent / "configs" / "example_config.yaml"
        
        if config_path.exists():
            config = Config(str(config_path))
        else:
            config = Config()
            
            # Set up embedder
            config.set('embedder.type', 'openai')
            config.set('embedder.model', 'text-embedding-ada-002')
            
            # Set up metadata store
            config.set('metadata_store.type', 'sqlite')
            config.set('metadata_store.database_path', './benchmark_metadata.db')
            
            # Set up settings
            config.set('settings.max_memories', 10000)
            config.set('settings.similarity_threshold', 0.6)
            config.set('settings.default_top_k', 5)
        
        # Override vector store settings for benchmarking
        config.set('vector_store.type', vector_store_type)
        
        if vector_store_type == 'faiss':
            config.set('vector_store.index_path', f'./benchmark_faiss_index')
        elif vector_store_type == 'qdrant':
            config.set('vector_store.url', 'http://localhost:6333')
            config.set('vector_store.collection_name', f'benchmark_collection_{int(time.time())}')
        
        config.set('vector_store.dimension', 1536)
        
        # Set API key if available
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  Using sentence transformers for demo...")
            config.set('embedder.type', 'sentence_transformers')
            config.set('embedder.model', 'all-MiniLM-L6-v2')
        else:
            config.set('embedder.api_key', api_key)
        
        return config
    
    def benchmark_vector_store(self, vector_store_type: str) -> Dict[str, Any]:
        """Benchmark a specific vector store."""
        print(f"\n🔧 Benchmarking {vector_store_type.upper()}...")
        
        try:
            # Create configuration
            config = self.create_config(vector_store_type)
            
            # Initialize memory API
            start_time = time.time()
            memory = MemoryAPI(config)
            init_time = time.time() - start_time
            
            # Store test data
            store_times = []
            memory_ids = []
            
            for item in self.test_data:
                start_time = time.time()
                memory_id = memory.store(item["content"], item["metadata"])
                store_time = time.time() - start_time
                
                store_times.append(store_time)
                memory_ids.append(memory_id)
            
            avg_store_time = sum(store_times) / len(store_times)
            total_store_time = sum(store_times)
            
            # Test retrieval
            retrieval_times = []
            retrieval_results = []
            
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
                retrieval_time = time.time() - start_time
                
                retrieval_times.append(retrieval_time)
                retrieval_results.append(results)
            
            avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
            total_retrieval_time = sum(retrieval_times)
            
            # Test metadata filtering
            filter_times = []
            
            for category in ["programming", "ai", "tools"]:
                start_time = time.time()
                all_results = memory.retrieve(
                    "technology",
                    top_k=15
                )
                # Filter results in Python
                results = [
                    result for result in all_results 
                    if result['metadata'].get('category') == category
                ][:5]
                filter_time = time.time() - start_time
                filter_times.append(filter_time)
            
            avg_filter_time = sum(filter_times) / len(filter_times)
            
            # Test update operations
            update_times = []
            
            for i, memory_id in enumerate(memory_ids[:3]):
                start_time = time.time()
                success = memory.update(
                    memory_id,
                    f"Updated content {i}",
                    {"updated": True, "timestamp": datetime.now().isoformat()}
                )
                update_time = time.time() - start_time
                update_times.append(update_time)
            
            avg_update_time = sum(update_times) / len(update_times) if update_times else 0
            
            # Test delete operations
            delete_times = []
            
            for memory_id in memory_ids[:3]:
                start_time = time.time()
                success = memory.delete(memory_id)
                delete_time = time.time() - start_time
                delete_times.append(delete_time)
            
            avg_delete_time = sum(delete_times) / len(delete_times) if delete_times else 0
            
            # Compile results
            results = {
                "vector_store_type": vector_store_type,
                "initialization_time": init_time,
                "store_operations": {
                    "total_time": total_store_time,
                    "average_time": avg_store_time,
                    "total_operations": len(self.test_data)
                },
                "retrieval_operations": {
                    "total_time": total_retrieval_time,
                    "average_time": avg_retrieval_time,
                    "total_operations": len(test_queries)
                },
                "filter_operations": {
                    "average_time": avg_filter_time,
                    "total_operations": len(filter_times)
                },
                "update_operations": {
                    "average_time": avg_update_time,
                    "total_operations": len(update_times)
                },
                "delete_operations": {
                    "average_time": avg_delete_time,
                    "total_operations": len(delete_times)
                },
                "success": True
            }
            
            print(f"✅ {vector_store_type.upper()} benchmark completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Error benchmarking {vector_store_type}: {e}")
            return {
                "vector_store_type": vector_store_type,
                "success": False,
                "error": str(e)
            }
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run comparison between vector stores."""
        print("🚀 Starting Vector Store Comparison")
        print("=" * 50)
        
        vector_stores = ["faiss", "qdrant"]
        
        for vs_type in vector_stores:
            self.results[vs_type] = self.benchmark_vector_store(vs_type)
        
        return self.results
    
    def print_comparison(self):
        """Print comparison results."""
        print("\n📊 Vector Store Comparison Results")
        print("=" * 50)
        
        # Check if we have successful results
        successful_results = {k: v for k, v in self.results.items() if v.get("success", False)}
        
        if not successful_results:
            print("❌ No successful benchmarks to compare.")
            return
        
        # Create comparison table
        print("\nPerformance Comparison:")
        print("-" * 80)
        print(f"{'Metric':<25} {'FAISS':<15} {'Qdrant':<15} {'Winner':<10}")
        print("-" * 80)
        
        metrics = [
            ("Initialization (s)", "initialization_time"),
            ("Avg Store (s)", "store_operations.average_time"),
            ("Avg Retrieval (s)", "retrieval_operations.average_time"),
            ("Avg Filter (s)", "filter_operations.average_time"),
            ("Avg Update (s)", "update_operations.average_time"),
            ("Avg Delete (s)", "delete_operations.average_time")
        ]
        
        for metric_name, metric_key in metrics:
            faiss_value = self._get_nested_value(self.results.get("faiss", {}), metric_key, 0)
            qdrant_value = self._get_nested_value(self.results.get("qdrant", {}), metric_key, 0)
            
            if faiss_value and qdrant_value:
                winner = "FAISS" if faiss_value < qdrant_value else "Qdrant"
                print(f"{metric_name:<25} {faiss_value:<15.4f} {qdrant_value:<15.4f} {winner:<10}")
            else:
                print(f"{metric_name:<25} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
        
        print("-" * 80)
        
        # Detailed analysis
        print("\n📈 Detailed Analysis:")
        print("-" * 50)
        
        for vs_type, result in successful_results.items():
            print(f"\n{vs_type.upper()} Results:")
            print(f"  Initialization: {result['initialization_time']:.4f}s")
            print(f"  Store Operations: {result['store_operations']['average_time']:.4f}s avg")
            print(f"  Retrieval Operations: {result['retrieval_operations']['average_time']:.4f}s avg")
            print(f"  Filter Operations: {result['filter_operations']['average_time']:.4f}s avg")
            print(f"  Update Operations: {result['update_operations']['average_time']:.4f}s avg")
            print(f"  Delete Operations: {result['delete_operations']['average_time']:.4f}s avg")
    
    def _get_nested_value(self, obj: Dict, key: str, default=None):
        """Get nested dictionary value using dot notation."""
        keys = key.split('.')
        value = obj
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def generate_recommendations(self):
        """Generate recommendations based on benchmark results."""
        print("\n💡 Recommendations")
        print("=" * 50)
        
        successful_results = {k: v for k, v in self.results.items() if v.get("success", False)}
        
        if not successful_results:
            print("❌ No successful benchmarks to generate recommendations from.")
            return
        
        print("\n🎯 Choose FAISS if you need:")
        print("  ✅ Fast local storage and retrieval")
        print("  ✅ Simple setup (no external dependencies)")
        print("  ✅ High performance for similarity search")
        print("  ✅ Offline operation")
        print("  ✅ Small to medium datasets")
        
        print("\n🎯 Choose Qdrant if you need:")
        print("  ✅ Advanced filtering and querying")
        print("  ✅ Scalability for large datasets")
        print("  ✅ Real-time updates and synchronization")
        print("  ✅ Complex metadata operations")
        print("  ✅ Multi-user access")
        print("  ✅ Persistence and backup features")
        
        # Performance-based recommendations
        if "faiss" in successful_results and "qdrant" in successful_results:
            faiss_retrieval = self._get_nested_value(self.results["faiss"], "retrieval_operations.average_time", 0)
            qdrant_retrieval = self._get_nested_value(self.results["qdrant"], "retrieval_operations.average_time", 0)
            
            if faiss_retrieval and qdrant_retrieval:
                if faiss_retrieval < qdrant_retrieval:
                    print(f"\n⚡ FAISS is {qdrant_retrieval/faiss_retrieval:.1f}x faster for retrieval")
                else:
                    print(f"\n⚡ Qdrant is {faiss_retrieval/qdrant_retrieval:.1f}x faster for retrieval")
        
        print("\n🔧 Setup Requirements:")
        print("  FAISS: pip install faiss-cpu (or faiss-gpu for GPU support)")
        print("  Qdrant: pip install qdrant-client + Qdrant server")
        
        print("\n📚 Use Cases:")
        print("  FAISS: Research, prototyping, single-user applications")
        print("  Qdrant: Production systems, multi-user applications, complex queries")


def demo_faiss_features():
    """Demonstrate FAISS-specific features."""
    print("\n🔧 FAISS Features Demo")
    print("=" * 50)
    
    config = Config()
    config.set('vector_store.type', 'faiss')
    config.set('vector_store.index_path', './faiss_demo_index')
    config.set('vector_store.dimension', 1536)
    
    # Set up embedder
    config.set('embedder.type', 'sentence_transformers')
    config.set('embedder.model', 'all-MiniLM-L6-v2')
    
    # Set up metadata store
    config.set('metadata_store.type', 'sqlite')
    config.set('metadata_store.database_path', './faiss_demo_metadata.db')
    
    try:
        memory = MemoryAPI(config)
        
        # Store some test data
        test_data = [
            ("Python programming language", {"category": "programming", "language": "python"}),
            ("Machine learning algorithms", {"category": "ai", "topic": "ml"}),
            ("Data structures and algorithms", {"category": "cs", "topic": "dsa"}),
            ("Web development frameworks", {"category": "web", "topic": "frameworks"}),
            ("Database management systems", {"category": "database", "topic": "dbms"})
        ]
        
        print("Storing test data in FAISS...")
        for content, metadata in test_data:
            memory.store(content, metadata)
        
        print("✅ FAISS demo completed!")
        
        # Test retrieval
        results = memory.retrieve("programming", top_k=3)
        print(f"Found {len(results)} results for 'programming'")
        
    except Exception as e:
        print(f"❌ FAISS demo error: {e}")


def demo_qdrant_features():
    """Demonstrate Qdrant-specific features."""
    print("\n🔧 Qdrant Features Demo")
    print("=" * 50)
    
    print("Note: This demo requires a running Qdrant server.")
    print("To start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    
    config = Config()
    config.set('vector_store.type', 'qdrant')
    config.set('vector_store.url', 'http://localhost:6333')
    config.set('vector_store.collection_name', 'demo_collection')
    config.set('vector_store.dimension', 1536)
    
    # Set up embedder
    config.set('embedder.type', 'sentence_transformers')
    config.set('embedder.model', 'all-MiniLM-L6-v2')
    
    # Set up metadata store
    config.set('metadata_store.type', 'sqlite')
    config.set('metadata_store.database_path', './qdrant_demo_metadata.db')
    
    try:
        memory = MemoryAPI(config)
        
        # Store some test data
        test_data = [
            ("Python programming language", {"category": "programming", "language": "python"}),
            ("Machine learning algorithms", {"category": "ai", "topic": "ml"}),
            ("Data structures and algorithms", {"category": "cs", "topic": "dsa"}),
            ("Web development frameworks", {"category": "web", "topic": "frameworks"}),
            ("Database management systems", {"category": "database", "topic": "dbms"})
        ]
        
        print("Storing test data in Qdrant...")
        for content, metadata in test_data:
            memory.store(content, metadata)
        
        print("✅ Qdrant demo completed!")
        
        # Test advanced filtering
        all_results = memory.retrieve(
            "technology",
            top_k=15
        )
        # Filter results in Python
        results = [
            result for result in all_results 
            if result['metadata'].get('category') == "programming"
        ][:5]
        print(f"Found {len(results)} programming-related results")
        
    except Exception as e:
        print(f"❌ Qdrant demo error: {e}")
        print("Make sure Qdrant server is running on http://localhost:6333")


def cleanup():
    """Clean up demo files."""
    print("\n🧹 Cleaning up demo files...")
    
    files_to_remove = [
        './benchmark_faiss_index',
        './benchmark_qdrant_index',
        './benchmark_faiss_metadata.db',
        './benchmark_qdrant_metadata.db',
        './faiss_demo_index',
        './faiss_demo_metadata.db',
        './qdrant_demo_metadata.db'
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
    parser = argparse.ArgumentParser(description='Memorix SDK Vector Store Comparison Demo')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode (no user interaction)')
    args = parser.parse_args()
    
    print("🔍 Memorix SDK - Vector Store Comparison Demo")
    print("=" * 50)
    print("This demo compares different vector stores")
    print("to help you choose the right one for your use case!")
    
    try:
        # Run benchmark comparison
        benchmark = VectorStoreBenchmark()
        results = benchmark.run_comparison()
        benchmark.print_comparison()
        benchmark.generate_recommendations()
        
        # Run feature demos
        demo_faiss_features()
        demo_qdrant_features()
        
        print("\n🎉 Vector store comparison demo completed successfully!")
        print("\nWhat you learned:")
        print("✅ How to compare different vector stores")
        print("✅ Performance characteristics of each store")
        print("✅ When to use FAISS vs Qdrant")
        print("✅ How to benchmark your own use cases")
        
        print("\n🚀 Ready to choose the best vector store for your application!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check your configuration and API keys.")
        return 1
    
    finally:
        # Ask user if they want to clean up (skip in test mode)
        if not args.test_mode:
            response = input("\n🧹 Clean up demo files? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                cleanup()
                print("✅ Cleanup completed!")
            else:
                print("💡 Demo files preserved for inspection.")
        else:
            # Auto-cleanup in test mode
            cleanup()
            print("✅ Test mode cleanup completed!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 