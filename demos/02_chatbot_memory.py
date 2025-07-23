#!/usr/bin/env python3
"""
Memorix SDK - Chatbot with Memory Demo
======================================

This demo shows how to build a conversational AI chatbot that:
- Remembers conversations and context
- Learns from user interactions
- Provides personalized responses
- Maintains conversation history

Perfect for building intelligent chatbots with persistent memory!
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add the parent directory to the path to import memorix
sys.path.append(str(Path(__file__).parent.parent.parent))

from memorix import MemoryAPI, Config


class ChatbotWithMemory:
    """A chatbot that uses Memorix for persistent memory."""
    
    def __init__(self, config: Config):
        """Initialize the chatbot with memory."""
        self.memory = MemoryAPI(config)
        self.conversation_history = []
        self.user_profiles = {}
        
    def store_conversation(self, user_id: str, message: str, response: str, 
                          context: Optional[Dict] = None):
        """Store a conversation exchange in memory."""
        # Store user message
        user_memory_id = self.memory.store(
            message,
            metadata={
                "type": "user_message",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "context": context or {}
            }
        )
        
        # Store bot response
        bot_memory_id = self.memory.store(
            response,
            metadata={
                "type": "bot_response",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "context": context or {},
                "related_message": user_memory_id
            }
        )
        
        # Store conversation context
        conversation_context = f"User: {message}\nBot: {response}"
        self.memory.store(
            conversation_context,
            metadata={
                "type": "conversation_context",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "user_message_id": user_memory_id,
                "bot_response_id": bot_memory_id
            }
        )
        
        return user_memory_id, bot_memory_id
    
    def get_user_context(self, user_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context for a user."""
        # Get recent conversations
        recent_context = self.memory.retrieve(
            query,
            top_k=top_k,
            metadata_filter={"user_id": user_id}
        )
        
        # Get general knowledge that might be relevant
        general_context = self.memory.retrieve(
            query,
            top_k=3,
            metadata_filter={"type": "knowledge"}
        )
        
        return recent_context + general_context
    
    def learn_from_interaction(self, user_id: str, message: str, response: str, 
                              user_feedback: Optional[str] = None):
        """Learn from user interactions and feedback."""
        # Store the interaction
        self.store_conversation(user_id, message, response)
        
        # If there's feedback, store it for learning
        if user_feedback:
            self.memory.store(
                f"Feedback on response to '{message}': {user_feedback}",
                metadata={
                    "type": "feedback",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "original_message": message,
                    "response": response,
                    "feedback": user_feedback
                }
            )
    
    def generate_response(self, user_id: str, message: str) -> str:
        """Generate a response based on memory and context."""
        # Get relevant context
        context = self.get_user_context(user_id, message)
        
        # Simple response generation based on context
        if context:
            # Use the most relevant context to generate response
            most_relevant = context[0]
            
            if most_relevant['metadata'].get('type') == 'bot_response':
                # If we have a similar previous response, adapt it
                return f"Based on our previous conversation, {most_relevant['content']}"
            elif most_relevant['metadata'].get('type') == 'knowledge':
                # Use knowledge to inform response
                return f"I remember that {most_relevant['content']}. How can I help you with that?"
            else:
                # Generic response with context
                return f"I see you're asking about this topic. Let me help you with that."
        else:
            # No relevant context found
            return "I'm here to help! What would you like to know?"
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get or create a user profile based on conversation history."""
        if user_id not in self.user_profiles:
            # Retrieve user's conversation history
            user_memories = self.memory.retrieve(
                "user preferences interests topics",
                top_k=10,
                metadata_filter={"user_id": user_id}
            )
            
            # Analyze user preferences
            topics = {}
            for memory in user_memories:
                content = memory['content'].lower()
                if 'python' in content:
                    topics['programming'] = topics.get('programming', 0) + 1
                if 'ai' in content or 'machine learning' in content:
                    topics['ai'] = topics.get('ai', 0) + 1
                if 'data' in content:
                    topics['data'] = topics.get('data', 0) + 1
            
            self.user_profiles[user_id] = {
                'user_id': user_id,
                'topics_of_interest': topics,
                'conversation_count': len(user_memories),
                'last_interaction': datetime.now().isoformat()
            }
        
        return self.user_profiles[user_id]


def setup_config():
    """Set up configuration for the chatbot demo."""
    print("🔧 Setting up chatbot configuration...")
    
    # Load configuration from YAML file
    config_path = Path(__file__).parent.parent / "configs" / "example_config.yaml"
    
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        config = Config(str(config_path))
    else:
        print("⚠️  Configuration file not found, using default config")
        config = Config()
        
        # Set up vector store
        config.set('vector_store.type', 'faiss')
        config.set('vector_store.index_path', './chatbot_index')
        config.set('vector_store.dimension', 1536)
        
        # Set up embedder
        config.set('embedder.type', 'openai')
        config.set('embedder.model', 'text-embedding-ada-002')
        
        # Set up metadata store
        config.set('metadata_store.type', 'sqlite')
        config.set('metadata_store.database_path', './chatbot_metadata.db')
        
        # Set up settings
        config.set('settings.max_memories', 10000)
        config.set('settings.similarity_threshold', 0.6)
        config.set('settings.default_top_k', 5)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Using sentence transformers for demo...")
        config.set('embedder.type', 'sentence_transformers')
        config.set('embedder.model', 'all-MiniLM-L6-v2')
    else:
        config.set('embedder.api_key', api_key)
    
    return config


def demo_conversation_flow(chatbot: ChatbotWithMemory):
    """Demonstrate a conversation flow with memory."""
    print("\n💬 Demo: Conversation Flow with Memory")
    print("=" * 50)
    
    user_id = "demo_user_001"
    
    # Simulate a conversation
    conversations = [
        ("Hi, I'm interested in learning Python", "Great! Python is an excellent programming language to start with. It's known for its simplicity and readability."),
        ("What are the best resources for beginners?", "For Python beginners, I recommend starting with the official Python tutorial, then moving to platforms like Codecademy or freeCodeCamp."),
        ("I also want to learn about machine learning", "Machine learning is a fascinating field! Since you're learning Python, you'll want to explore libraries like scikit-learn, TensorFlow, and PyTorch."),
        ("Can you remind me what we talked about earlier?", "Of course! We discussed Python programming for beginners, learning resources, and your interest in machine learning."),
        ("What's the best way to practice Python?", "The best way to practice Python is through hands-on projects. Start with simple exercises, then build small applications like calculators or games.")
    ]
    
    for i, (message, response) in enumerate(conversations, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"User: {message}")
        print(f"Bot: {response}")
        
        # Store the conversation
        chatbot.learn_from_interaction(user_id, message, response)
        
        # Small delay for demo effect
        time.sleep(1)
    
    print("\n✅ Conversation stored in memory!")


def demo_memory_retrieval(chatbot: ChatbotWithMemory):
    """Demonstrate memory retrieval capabilities."""
    print("\n🧠 Demo: Memory Retrieval")
    print("=" * 50)
    
    user_id = "demo_user_001"
    
    # Test different types of queries
    test_queries = [
        "Python programming",
        "machine learning",
        "learning resources",
        "practice exercises"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        context = chatbot.get_user_context(user_id, query, top_k=3)
        
        if context:
            print(f"Found {len(context)} relevant memories:")
            for i, memory in enumerate(context, 1):
                print(f"  {i}. {memory['content'][:60]}...")
                print(f"     Type: {memory['metadata'].get('type', 'unknown')}")
                print(f"     Similarity: {memory['similarity']:.3f}")
        else:
            print("  No relevant memories found.")


def demo_user_profiles(chatbot: ChatbotWithMemory):
    """Demonstrate user profile generation."""
    print("\n👤 Demo: User Profile Generation")
    print("=" * 50)
    
    user_id = "demo_user_001"
    
    # Get user profile
    profile = chatbot.get_user_profile(user_id)
    
    print(f"User Profile for {user_id}:")
    print(f"  Topics of Interest: {profile['topics_of_interest']}")
    print(f"  Conversation Count: {profile['conversation_count']}")
    print(f"  Last Interaction: {profile['last_interaction']}")
    
    # Show how this can be used for personalization
    if profile['topics_of_interest'].get('programming', 0) > 0:
        print("  🎯 This user is interested in programming!")
    if profile['topics_of_interest'].get('ai', 0) > 0:
        print("  🤖 This user is interested in AI and machine learning!")


def demo_interactive_chat(chatbot: ChatbotWithMemory):
    """Demonstrate interactive chat functionality."""
    print("\n🎮 Demo: Interactive Chat")
    print("=" * 50)
    print("You can now chat with the bot! Type 'quit' to exit.")
    print("The bot will remember our conversation and use it for context.")
    
    user_id = "interactive_user"
    
    while True:
        try:
            message = input("\nYou: ").strip()
            
            if message.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye! I'll remember our conversation for next time.")
                break
            
            if not message:
                continue
            
            # Generate response
            response = chatbot.generate_response(user_id, message)
            print(f"Bot: {response}")
            
            # Store the interaction
            chatbot.learn_from_interaction(user_id, message, response)
            
            # Ask for feedback (optional)
            feedback = input("Was this response helpful? (y/n, or press Enter to skip): ").strip()
            if feedback.lower() in ['y', 'yes']:
                chatbot.learn_from_interaction(user_id, message, response, "positive")
            elif feedback.lower() in ['n', 'no']:
                chatbot.learn_from_interaction(user_id, message, response, "negative")
            
        except KeyboardInterrupt:
            print("\nBot: Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"Bot: Sorry, I encountered an error: {e}")


def demo_knowledge_base(chatbot: ChatbotWithMemory):
    """Demonstrate knowledge base integration."""
    print("\n📚 Demo: Knowledge Base Integration")
    print("=" * 50)
    
    # Store some knowledge
    knowledge_items = [
        {
            "content": "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.",
            "metadata": {"type": "knowledge", "topic": "python", "category": "programming"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "metadata": {"type": "knowledge", "topic": "machine_learning", "category": "ai"}
        },
        {
            "content": "The best way to learn programming is through practice and building real projects.",
            "metadata": {"type": "knowledge", "topic": "learning", "category": "education"}
        },
        {
            "content": "Git is a distributed version control system that helps track changes in code.",
            "metadata": {"type": "knowledge", "topic": "git", "category": "tools"}
        }
    ]
    
    print("Storing knowledge base...")
    for item in knowledge_items:
        chatbot.memory.store(item["content"], item["metadata"])
    
    print("✅ Knowledge base created!")
    
    # Test knowledge retrieval
    print("\nTesting knowledge retrieval...")
    test_query = "What is Python?"
    context = chatbot.get_user_context("any_user", test_query, top_k=3)
    
    if context:
        print(f"Found knowledge for '{test_query}':")
        for memory in context:
            if memory['metadata'].get('type') == 'knowledge':
                print(f"  📖 {memory['content']}")


def cleanup():
    """Clean up demo files."""
    print("\n🧹 Cleaning up demo files...")
    
    files_to_remove = [
        './chatbot_index',
        './chatbot_metadata.db'
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
    parser = argparse.ArgumentParser(description='Memorix SDK Chatbot Memory Demo')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode (no user interaction)')
    args = parser.parse_args()
    
    print("🤖 Memorix SDK - Chatbot with Memory Demo")
    print("=" * 50)
    print("This demo shows how to build a conversational AI")
    print("with persistent memory using Memorix SDK!")
    
    try:
        # Set up configuration
        config = setup_config()
        
        # Initialize chatbot
        print("\n🚀 Initializing Chatbot with Memory...")
        chatbot = ChatbotWithMemory(config)
        print("✅ Chatbot initialized successfully!")
        
        # Run demos
        demo_knowledge_base(chatbot)
        demo_conversation_flow(chatbot)
        demo_memory_retrieval(chatbot)
        demo_user_profiles(chatbot)
        
        # Interactive demo (skip in test mode)
        if not args.test_mode:
            print("\n" + "="*50)
            print("🎮 INTERACTIVE DEMO")
            print("="*50)
            demo_interactive_chat(chatbot)
        
        print("\n🎉 Chatbot demo completed successfully!")
        print("\nWhat you learned:")
        print("✅ How to build chatbots with persistent memory")
        print("✅ How to store and retrieve conversation context")
        print("✅ How to generate user profiles")
        print("✅ How to integrate knowledge bases")
        print("✅ How to learn from user feedback")
        
        print("\n🚀 Ready to build your own intelligent chatbot!")
        
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