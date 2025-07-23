#!/usr/bin/env python3
"""
Memorix SDK - Basic Usage Demo
==============================

This demo shows the fundamental features of the Memorix SDK:
- Memory storage and retrieval
- Metadata management
- Similarity search
- Basic CRUD operations

Run this demo to get started with Memorix in under 5 minutes!
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import memorix
sys.path.append(str(Path(__file__).parent.parent.parent))

from memorix import MemoryAPI, Config


def setup_config():
    """Set up configuration for the demo."""
    print("🔧 Setting up configuration...")
    
    # Load configuration from YAML file
    config_path = Path(__file__).parent.parent / "configs" / "example_config.yaml"
    
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        config = Config(str(config_path))
    else:
        print("⚠️  Configuration file not found, using default config")
        config = Config()
        
        # Set up vector store (FAISS for this demo)
        config.set('vector_store.type', 'faiss')
        config.set('vector_store.index_path', './demo_index')
        config.set('vector_store.dimension', 1536)
        
        # Set up embedder (OpenAI)
        config.set('embedder.type', 'openai')
        config.set('embedder.model', 'text-embedding-ada-002')
        
        # Set up metadata store
        config.set('metadata_store.type', 'sqlite')
        config.set('metadata_store.database_path', './demo_metadata.db')
        
        # Set up settings
        config.set('settings.max_memories', 1000)
        config.set('settings.similarity_threshold', 0.7)
        config.set('settings.default_top_k', 5)
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   Using sentence transformers for demo purposes...")
        # For demo purposes, we'll use sentence transformers
        config.set('embedder.type', 'sentence_transformers')
        config.set('embedder.model', 'all-MiniLM-L6-v2')
    else:
        config.set('embedder.api_key', api_key)
    
    return config


def demo_basic_operations(memory):
    """Demonstrate basic memory operations."""
    print("\n📝 Demo 1: Basic Memory Operations")
    print("=" * 50)
    
    # Store some memories
    print("Storing memories...")
    
    memories = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "metadata": {"topic": "programming", "language": "python", "difficulty": "beginner"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "metadata": {"topic": "AI", "category": "machine learning", "difficulty": "intermediate"}
        },
        {
            "content": "Data structures are ways of organizing and storing data for efficient access and modification.",
            "metadata": {"topic": "computer science", "category": "data structures", "difficulty": "intermediate"}
        },
        {
            "content": "Neural networks are computing systems inspired by biological neural networks.",
            "metadata": {"topic": "AI", "category": "neural networks", "difficulty": "advanced"}
        },
        {
            "content": "Git is a distributed version control system for tracking changes in source code.",
            "metadata": {"topic": "programming", "category": "version control", "difficulty": "beginner"}
        }
    ]
    
    memory_ids = []
    for i, mem in enumerate(memories, 1):
        memory_id = memory.store(mem["content"], mem["metadata"])
        memory_ids.append(memory_id)
        print(f"  {i}. Stored memory {memory_id[:8]}...")
    
    print(f"✅ Successfully stored {len(memories)} memories!")
    return memory_ids


def demo_retrieval(memory):
    """Demonstrate memory retrieval."""
    print("\n🔍 Demo 2: Memory Retrieval")
    print("=" * 50)
    
    # Test queries
    queries = [
        "programming languages",
        "artificial intelligence",
        "data organization",
        "version control systems"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = memory.retrieve(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['content'][:80]}...")
                print(f"     Similarity: {result['similarity']:.3f}")
                print(f"     Metadata: {result['metadata']}")
        else:
            print("  No relevant memories found.")


def demo_metadata_filtering(memory):
    """Demonstrate metadata-based filtering."""
    print("\n🏷️  Demo 3: Metadata Filtering")
    print("=" * 50)
    
    # Search for AI-related content with difficulty filter
    print("Searching for AI content with intermediate difficulty...")
    results = memory.retrieve(
        "artificial intelligence",
        top_k=5,
        metadata_filter={"difficulty": "intermediate"}
    )
    
    if results:
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['content'][:60]}...")
            print(f"     Difficulty: {result['metadata'].get('difficulty', 'N/A')}")
    else:
        print("No results found with the specified criteria.")


def demo_update_and_delete(memory, memory_ids):
    """Demonstrate update and delete operations."""
    print("\n🔄 Demo 4: Update and Delete Operations")
    print("=" * 50)
    
    # Update a memory
    memory_id = memory_ids[0]
    print(f"Updating memory {memory_id[:8]}...")
    
    success = memory.update(
        memory_id,
        "Python is a high-level, interpreted programming language known for its simplicity, readability, and extensive standard library.",
        metadata={"topic": "programming", "language": "python", "difficulty": "beginner", "updated": True}
    )
    
    if success:
        print("✅ Memory updated successfully!")
        
        # Retrieve the updated memory
        results = memory.retrieve("python programming", top_k=1)
        if results:
            print(f"Updated content: {results[0]['content']}")
            print(f"Updated metadata: {results[0]['metadata']}")
    
    # Delete a memory
    memory_id_to_delete = memory_ids[-1]
    print(f"\nDeleting memory {memory_id_to_delete[:8]}...")
    
    success = memory.delete(memory_id_to_delete)
    if success:
        print("✅ Memory deleted successfully!")
        
        # Verify deletion
        memories = memory.list_memories(limit=10)
        print(f"Remaining memories: {len(memories)}")


def demo_list_memories(memory):
    """Demonstrate listing memories."""
    print("\n📋 Demo 5: Listing Memories")
    print("=" * 50)
    
    memories = memory.list_memories(limit=10)
    
    if memories:
        print(f"Found {len(memories)} memories:")
        for i, mem in enumerate(memories, 1):
            print(f"  {i}. ID: {mem['memory_id'][:8]}...")
            print(f"     Content: {mem['content'][:60]}...")
            print(f"     Metadata: {mem['metadata']}")
            print()
    else:
        print("No memories found.")


def demo_similarity_threshold(memory):
    """Demonstrate similarity threshold effects."""
    print("\n🎯 Demo 6: Similarity Threshold")
    print("=" * 50)
    
    query = "programming"
    
    # Test different similarity thresholds
    thresholds = [0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        print(f"\nSearching with similarity threshold: {threshold}")
        results = memory.retrieve(query, top_k=5, similarity_threshold=threshold)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Similarity: {result['similarity']:.3f}")
                print(f"     {result['content'][:50]}...")
        else:
            print("No results found above the threshold.")


def cleanup():
    """Clean up demo files."""
    print("\n🧹 Cleaning up demo files...")
    
    files_to_remove = [
        './demo_index',
        './demo_metadata.db',
        './demo_metadata.json'
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
    parser = argparse.ArgumentParser(description='Memorix SDK Basic Usage Demo')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode (no user interaction)')
    args = parser.parse_args()
    
    print("🧠 Memorix SDK - Basic Usage Demo")
    print("=" * 50)
    print("This demo will show you the core features of Memorix SDK")
    print("in under 5 minutes!")
    
    try:
        # Set up configuration
        config = setup_config()
        
        # Initialize memory API
        print("\n🚀 Initializing Memorix...")
        memory = MemoryAPI(config)
        print("✅ Memorix initialized successfully!")
        
        # Run demos
        memory_ids = demo_basic_operations(memory)
        demo_retrieval(memory)
        demo_metadata_filtering(memory)
        demo_update_and_delete(memory, memory_ids)
        demo_list_memories(memory)
        demo_similarity_threshold(memory)
        
        print("\n🎉 Demo completed successfully!")
        print("\nWhat you learned:")
        print("✅ How to store and retrieve memories")
        print("✅ How to use metadata for filtering")
        print("✅ How to update and delete memories")
        print("✅ How similarity thresholds work")
        print("✅ Basic CRUD operations")
        
        print("\n🚀 Ready to build your own AI applications with memory!")
        
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