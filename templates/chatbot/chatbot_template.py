#!/usr/bin/env python3
"""
Memorix SDK - Chatbot Template
==============================

A ready-to-use chatbot template with memory capabilities.
Customize this template for your own chatbot applications!

Features:
- Persistent conversation memory
- User context tracking
- Knowledge base integration
- Customizable response generation
- Session management
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add the parent directory to the path to import memorix
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from memorix import MemoryAPI, Config


class ChatbotTemplate:
    """A customizable chatbot template with memory."""
    
    def __init__(self, config: Config, name: str = "Assistant"):
        """Initialize the chatbot."""
        self.memory = MemoryAPI(config)
        self.name = name
        self.conversation_history = []
        self.user_sessions = {}
        
        # Load knowledge base if available
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base from file if available."""
        knowledge_file = Path(__file__).parent / "knowledge_base.json"
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load knowledge base: {e}")
        return {}
    
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
        conversation_context = f"User: {message}\n{self.name}: {response}"
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
        all_context = self.memory.retrieve(
            query,
            top_k=top_k + 3
        )
        
        # Filter for user-specific context
        recent_context = [
            result for result in all_context 
            if result['metadata'].get('user_id') == user_id
        ][:top_k]
        
        # Filter for general knowledge
        general_context = [
            result for result in all_context 
            if result['metadata'].get('type') == 'knowledge'
        ][:3]
        
        return recent_context + general_context
    
    def generate_response(self, user_id: str, message: str) -> str:
        """Generate a response based on memory and context."""
        # Get relevant context
        context = self.get_user_context(user_id, message)
        
        # Check knowledge base first
        kb_response = self._check_knowledge_base(message)
        if kb_response:
            return kb_response
        
        # Generate response based on context
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
            return f"I'm {self.name}, and I'm here to help! What would you like to know?"
    
    def _check_knowledge_base(self, message: str) -> Optional[str]:
        """Check knowledge base for relevant information."""
        if not self.knowledge_base:
            return None
        
        # Simple keyword matching (you can enhance this)
        message_lower = message.lower()
        
        for topic, info in self.knowledge_base.items():
            if topic.lower() in message_lower:
                return info.get('response', f"I know about {topic}. What would you like to know?")
        
        return None
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get or create a user profile based on conversation history."""
        if user_id not in self.user_sessions:
            # Retrieve user's conversation history
            all_memories = self.memory.retrieve(
                "user preferences interests topics",
                top_k=20
            )
            
            # Filter for user-specific memories
            user_memories = [
                memory for memory in all_memories 
                if memory['metadata'].get('user_id') == user_id
            ][:10]
            
            # Analyze user preferences
            topics = {}
            for memory in user_memories:
                content = memory['content'].lower()
                # Add your own topic detection logic here
                if 'python' in content:
                    topics['programming'] = topics.get('programming', 0) + 1
                if 'ai' in content or 'machine learning' in content:
                    topics['ai'] = topics.get('ai', 0) + 1
                if 'data' in content:
                    topics['data'] = topics.get('data', 0) + 1
            
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'topics_of_interest': topics,
                'conversation_count': len(user_memories),
                'last_interaction': datetime.now().isoformat()
            }
        
        return self.user_sessions[user_id]
    
    def add_knowledge(self, topic: str, content: str, response: str):
        """Add knowledge to the chatbot."""
        # Store in memory
        self.memory.store(
            content,
            metadata={
                "type": "knowledge",
                "topic": topic,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Update knowledge base
        self.knowledge_base[topic] = {
            "content": content,
            "response": response
        }
    
    def chat(self, user_id: str, message: str) -> str:
        """Main chat method."""
        # Generate response
        response = self.generate_response(user_id, message)
        
        # Store conversation
        self.store_conversation(user_id, message, response)
        
        # Update user profile
        self.get_user_profile(user_id)
        
        return response


def create_default_config() -> Config:
    """Create a default configuration for the chatbot."""
    config = Config()
    
    # Set up vector store
    config.set('vector_store.type', 'faiss')
    config.set('vector_store.index_path', './chatbot_index')
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
    config.set('metadata_store.database_path', './chatbot_metadata.db')
    
    # Set up settings
    config.set('settings.max_memories', 10000)
    config.set('settings.similarity_threshold', 0.6)
    config.set('settings.default_top_k', 5)
    
    return config


def create_sample_knowledge_base() -> Dict[str, Any]:
    """Create a sample knowledge base."""
    return {
        "python": {
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "response": "Python is a great programming language! It's known for its simple syntax and extensive libraries. What would you like to know about Python?"
        },
        "machine learning": {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "response": "Machine learning is fascinating! It allows computers to learn patterns from data. Are you interested in getting started with ML?"
        },
        "data science": {
            "content": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
            "response": "Data science is a powerful field that helps us understand data! What aspect of data science interests you?"
        }
    }


def main():
    """Example usage of the chatbot template."""
    print("🤖 Memorix SDK - Chatbot Template")
    print("=" * 50)
    print("This is a customizable chatbot template with memory capabilities.")
    
    # Create configuration
    config = create_default_config()
    
    # Initialize chatbot
    chatbot = ChatbotTemplate(config, name="MemorixBot")
    
    # Add sample knowledge
    knowledge_base = create_sample_knowledge_base()
    for topic, info in knowledge_base.items():
        chatbot.add_knowledge(topic, info["content"], info["response"])
    
    print("✅ Chatbot initialized with sample knowledge!")
    print("\nYou can now chat with the bot. Type 'quit' to exit.")
    print("The bot will remember our conversation and use it for context.")
    
    user_id = "demo_user"
    
    while True:
        try:
            message = input("\nYou: ").strip()
            
            if message.lower() in ['quit', 'exit', 'bye']:
                print(f"{chatbot.name}: Goodbye! I'll remember our conversation for next time.")
                break
            
            if not message:
                continue
            
            # Generate response
            response = chatbot.chat(user_id, message)
            print(f"{chatbot.name}: {response}")
            
        except KeyboardInterrupt:
            print(f"\n{chatbot.name}: Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"{chatbot.name}: Sorry, I encountered an error: {e}")


if __name__ == "__main__":
    main() 