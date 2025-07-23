#!/usr/bin/env python3
"""
Memorix SDK - Streamlit Web App
===============================

A beautiful web interface for exploring and interacting with Memorix SDK.
Features:
- Memory storage and retrieval
- Real-time search
- Metadata filtering
- Visual analytics
- Interactive demos

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add the parent directory to the path to import memorix
sys.path.append(str(Path(__file__).parent.parent.parent))

from memorix import MemoryAPI, Config


# Page configuration
st.set_page_config(
    page_title="Memorix SDK Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_memory():
    """Initialize the memory system with caching."""
    config = Config()
    
    # Set up vector store
    config.set('vector_store.type', 'faiss')
    config.set('vector_store.index_path', './streamlit_index')
    config.set('vector_store.dimension', 1536)
    
    # Set up embedder
    config.set('embedder.type', 'openai')
    config.set('embedder.model', 'text-embedding-ada-002')
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.warning("⚠️ OpenAI API key not found. Using sentence transformers for demo.")
        config.set('embedder.type', 'sentence_transformers')
        config.set('embedder.model', 'all-MiniLM-L6-v2')
    else:
        config.set('embedder.api_key', api_key)
    
    # Set up metadata store
    config.set('metadata_store.type', 'sqlite')
    config.set('metadata_store.database_path', './streamlit_metadata.db')
    
    # Set up settings
    config.set('settings.max_memories', 10000)
    config.set('settings.similarity_threshold', 0.6)
    config.set('settings.default_top_k', 10)
    
    return MemoryAPI(config)


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Memorix SDK Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Memory Management System")
    
    # Initialize memory system
    try:
        memory = initialize_memory()
        st.success("✅ Memory system initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize memory system: {e}")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "📝 Store Memories", "🔍 Search & Retrieve", "📊 Analytics", "🎮 Interactive Demo", "⚙️ Settings"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard(memory)
    elif page == "📝 Store Memories":
        show_store_memories(memory)
    elif page == "🔍 Search & Retrieve":
        show_search_retrieve(memory)
    elif page == "📊 Analytics":
        show_analytics(memory)
    elif page == "🎮 Interactive Demo":
        show_interactive_demo(memory)
    elif page == "⚙️ Settings":
        show_settings(memory)


def show_dashboard(memory):
    """Show the main dashboard."""
    st.header("📊 Dashboard")
    
    # Get basic statistics
    try:
        memories = memory.list_memories(limit=1000)
        total_memories = len(memories)
        
        # Calculate statistics
        if memories:
            # Extract metadata for analysis
            categories = {}
            tags = {}
            recent_memories = []
            
            for mem in memories:
                metadata = mem.get('metadata', {})
                
                # Count categories
                category = metadata.get('category', 'uncategorized')
                categories[category] = categories.get(category, 0) + 1
                
                # Count tags
                for tag in metadata.get('tags', []):
                    tags[tag] = tags.get(tag, 0) + 1
                
                # Recent memories
                if 'timestamp' in metadata:
                    recent_memories.append(mem)
            
            # Sort recent memories by timestamp
            recent_memories.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Memories", total_memories)
            
            with col2:
                st.metric("Categories", len(categories))
            
            with col3:
                st.metric("Unique Tags", len(tags))
            
            with col4:
                avg_length = sum(len(mem.get('content', '')) for mem in memories) / total_memories if total_memories > 0 else 0
                st.metric("Avg Content Length", f"{avg_length:.0f} chars")
            
            # Recent activity
            st.subheader("🕒 Recent Activity")
            if recent_memories:
                recent_df = pd.DataFrame([
                    {
                        'Content': mem.get('content', '')[:50] + '...',
                        'Category': mem.get('metadata', {}).get('category', 'N/A'),
                        'Tags': ', '.join(mem.get('metadata', {}).get('tags', [])),
                        'Timestamp': mem.get('metadata', {}).get('timestamp', 'N/A')
                    }
                    for mem in recent_memories[:10]
                ])
                st.dataframe(recent_df, use_container_width=True)
            
            # Category distribution
            if categories:
                st.subheader("📈 Category Distribution")
                fig = px.pie(
                    values=list(categories.values()),
                    names=list(categories.keys()),
                    title="Memory Distribution by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tag cloud
            if tags:
                st.subheader("🏷️ Popular Tags")
                tag_df = pd.DataFrame([
                    {'Tag': tag, 'Count': count}
                    for tag, count in sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10]
                ])
                fig = px.bar(tag_df, x='Tag', y='Count', title="Top 10 Tags")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("📝 No memories stored yet. Start by adding some memories!")
            
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")


def show_store_memories(memory):
    """Show the memory storage interface."""
    st.header("📝 Store Memories")
    
    # Single memory storage
    st.subheader("Add Single Memory")
    
    with st.form("single_memory_form"):
        content = st.text_area("Memory Content", height=150, placeholder="Enter the content you want to store...")
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.text_input("Category", placeholder="e.g., programming, ai, personal")
            tags = st.text_input("Tags (comma-separated)", placeholder="e.g., python, tutorial, beginner")
        
        with col2:
            difficulty = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced", "expert"])
            priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
        
        additional_metadata = st.text_area("Additional Metadata (JSON)", height=100, 
                                         placeholder='{"source": "book", "author": "John Doe"}')
        
        submitted = st.form_submit_button("Store Memory")
        
        if submitted and content:
            try:
                # Prepare metadata
                metadata = {
                    "category": category,
                    "difficulty": difficulty,
                    "priority": priority,
                    "timestamp": datetime.now().isoformat()
                }
                
                if tags:
                    metadata["tags"] = [tag.strip() for tag in tags.split(",")]
                
                if additional_metadata:
                    try:
                        additional = json.loads(additional_metadata)
                        metadata.update(additional)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in additional metadata")
                        return
                
                # Store memory
                memory_id = memory.store(content, metadata)
                
                st.success(f"✅ Memory stored successfully! ID: {memory_id[:8]}...")
                
                # Show stored memory
                with st.expander("View Stored Memory"):
                    st.json({
                        "memory_id": memory_id,
                        "content": content,
                        "metadata": metadata
                    })
            
            except Exception as e:
                st.error(f"❌ Error storing memory: {e}")
    
    # Batch memory storage
    st.subheader("Batch Memory Storage")
    
    with st.form("batch_memory_form"):
        batch_content = st.text_area(
            "Batch Content (one memory per line)",
            height=200,
            placeholder="Memory 1 content\nMemory 2 content\nMemory 3 content"
        )
        
        batch_category = st.text_input("Default Category", placeholder="e.g., programming")
        batch_tags = st.text_input("Default Tags (comma-separated)", placeholder="e.g., tutorial, demo")
        
        batch_submitted = st.form_submit_button("Store Batch")
        
        if batch_submitted and batch_content:
            try:
                memories = [line.strip() for line in batch_content.split('\n') if line.strip()]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                stored_count = 0
                for i, mem_content in enumerate(memories):
                    metadata = {
                        "category": batch_category,
                        "tags": [tag.strip() for tag in batch_tags.split(",")] if batch_tags else [],
                        "timestamp": datetime.now().isoformat(),
                        "batch_import": True
                    }
                    
                    memory.store(mem_content, metadata)
                    stored_count += 1
                    
                    progress_bar.progress((i + 1) / len(memories))
                    status_text.text(f"Stored {i + 1} of {len(memories)} memories...")
                
                st.success(f"✅ Successfully stored {stored_count} memories!")
                
            except Exception as e:
                st.error(f"❌ Error in batch storage: {e}")


def show_search_retrieve(memory):
    """Show the search and retrieve interface."""
    st.header("🔍 Search & Retrieve")
    
    # Search interface
    st.subheader("Search Memories")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Search Query", placeholder="Enter your search query...")
        top_k = st.slider("Number of Results", min_value=1, max_value=20, value=5)
    
    with col2:
        similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    
    # Metadata filters
    st.subheader("Metadata Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_category = st.text_input("Filter by Category", placeholder="e.g., programming")
    
    with col2:
        filter_difficulty = st.selectbox("Filter by Difficulty", ["", "beginner", "intermediate", "advanced", "expert"])
    
    with col3:
        filter_tags = st.text_input("Filter by Tags", placeholder="e.g., python, tutorial")
    
    # Build metadata filter
    metadata_filter = {}
    if filter_category:
        metadata_filter["category"] = filter_category
    if filter_difficulty:
        metadata_filter["difficulty"] = filter_difficulty
    if filter_tags:
        metadata_filter["tags"] = [tag.strip() for tag in filter_tags.split(",")]
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if query:
            try:
                with st.spinner("Searching..."):
                    results = memory.retrieve(
                        query,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        metadata_filter=metadata_filter if metadata_filter else None
                    )
                
                if results:
                    st.success(f"✅ Found {len(results)} results")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} (Similarity: {result['similarity']:.3f})"):
                            st.write("**Content:**")
                            st.write(result['content'])
                            
                            st.write("**Metadata:**")
                            st.json(result['metadata'])
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Update {i}", key=f"update_{i}"):
                                    st.session_state.editing_result = i
                                    st.session_state.editing_content = result['content']
                                    st.session_state.editing_metadata = result['metadata']
                            
                            with col2:
                                if st.button(f"Delete {i}", key=f"delete_{i}"):
                                    try:
                                        memory.delete(result['memory_id'])
                                        st.success(f"✅ Memory {result['memory_id'][:8]}... deleted")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ Error deleting memory: {e}")
                else:
                    st.info("🔍 No results found matching your criteria.")
            
            except Exception as e:
                st.error(f"❌ Error during search: {e}")
        else:
            st.warning("⚠️ Please enter a search query.")
    
    # Quick search examples
    st.subheader("Quick Search Examples")
    
    examples = [
        "Python programming",
        "Machine learning algorithms",
        "Data structures",
        "Web development",
        "Artificial intelligence"
    ]
    
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.quick_query = example
                st.rerun()
    
    # Check if we have a quick query
    if hasattr(st.session_state, 'quick_query'):
        query = st.session_state.quick_query
        del st.session_state.quick_query
        
        # Auto-execute the search
        try:
            results = memory.retrieve(query, top_k=5)
            if results:
                st.success(f"✅ Quick search for '{query}' found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Quick Result {i} (Similarity: {result['similarity']:.3f})"):
                        st.write(result['content'])
                        st.json(result['metadata'])
        except Exception as e:
            st.error(f"❌ Error in quick search: {e}")


def show_analytics(memory):
    """Show analytics and visualizations."""
    st.header("📊 Analytics")
    
    try:
        memories = memory.list_memories(limit=1000)
        
        if not memories:
            st.info("📝 No memories to analyze. Add some memories first!")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'content': mem.get('content', ''),
                'content_length': len(mem.get('content', '')),
                'category': mem.get('metadata', {}).get('category', 'uncategorized'),
                'difficulty': mem.get('metadata', {}).get('difficulty', 'unknown'),
                'priority': mem.get('metadata', {}).get('priority', 'unknown'),
                'tags': mem.get('metadata', {}).get('tags', []),
                'timestamp': mem.get('metadata', {}).get('timestamp', ''),
                'memory_id': mem.get('memory_id', '')
            }
            for mem in memories
        ])
        
        # Basic statistics
        st.subheader("📈 Basic Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Memories", len(df))
        
        with col2:
            avg_length = df['content_length'].mean()
            st.metric("Average Content Length", f"{avg_length:.0f} chars")
        
        with col3:
            unique_categories = df['category'].nunique()
            st.metric("Unique Categories", unique_categories)
        
        with col4:
            total_tags = sum(len(tags) for tags in df['tags'] if tags)
            st.metric("Total Tags", total_tags)
        
        # Content length distribution
        st.subheader("📏 Content Length Distribution")
        fig = px.histogram(df, x='content_length', nbins=20, title="Distribution of Content Lengths")
        st.plotly_chart(fig, use_container_width=True)
        
        # Category analysis
        st.subheader("📂 Category Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, title="Memory Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=category_counts.index, y=category_counts.values, title="Category Counts")
            st.plotly_chart(fig, use_container_width=True)
        
        # Difficulty analysis
        st.subheader("🎯 Difficulty Analysis")
        
        difficulty_counts = df['difficulty'].value_counts()
        fig = px.bar(x=difficulty_counts.index, y=difficulty_counts.values, title="Memory Distribution by Difficulty")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tag analysis
        st.subheader("🏷️ Tag Analysis")
        
        # Flatten tags
        all_tags = []
        for tags in df['tags']:
            if tags:
                all_tags.extend(tags)
        
        if all_tags:
            tag_counts = pd.Series(all_tags).value_counts().head(10)
            fig = px.bar(x=tag_counts.index, y=tag_counts.values, title="Top 10 Tags")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis (if timestamps are available)
        st.subheader("⏰ Time Series Analysis")
        
        # Filter memories with timestamps
        time_df = df[df['timestamp'] != ''].copy()
        
        if not time_df.empty:
            try:
                time_df['datetime'] = pd.to_datetime(time_df['timestamp'])
                time_df['date'] = time_df['datetime'].dt.date
                
                daily_counts = time_df.groupby('date').size().reset_index(name='count')
                daily_counts['date'] = pd.to_datetime(daily_counts['date'])
                
                fig = px.line(daily_counts, x='date', y='count', title="Memories Added Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not parse timestamps: {e}")
        
        # Content analysis
        st.subheader("📝 Content Analysis")
        
        # Word count analysis
        df['word_count'] = df['content'].str.split().str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='word_count', nbins=20, title="Word Count Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Most common words (simple analysis)
            all_words = ' '.join(df['content']).lower().split()
            word_counts = pd.Series(all_words).value_counts().head(10)
            fig = px.bar(x=word_counts.index, y=word_counts.values, title="Most Common Words")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error loading analytics: {e}")


def show_interactive_demo(memory):
    """Show interactive demo features."""
    st.header("🎮 Interactive Demo")
    
    st.markdown("""
    This interactive demo lets you explore Memorix SDK features in real-time.
    Try different queries and see how the memory system responds!
    """)
    
    # Chat-like interface
    st.subheader("💬 Chat with Memory")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Search for relevant memories
                    results = memory.retrieve(prompt, top_k=3)
                    
                    if results:
                        response = f"I found some relevant information:\n\n"
                        for i, result in enumerate(results, 1):
                            response += f"**{i}.** {result['content']}\n"
                            response += f"*Similarity: {result['similarity']:.3f}*\n\n"
                        
                        # Store the interaction
                        memory.store(
                            f"User asked: {prompt}",
                            metadata={
                                "type": "chat_interaction",
                                "timestamp": datetime.now().isoformat(),
                                "category": "chat"
                            }
                        )
                    else:
                        response = "I don't have any specific information about that, but I'm here to help! Try asking about programming, AI, or other topics I might know about."
                    
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.write(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Memory exploration
    st.subheader("🔍 Memory Explorer")
    
    # Random memory display
    if st.button("Show Random Memory"):
        try:
            memories = memory.list_memories(limit=100)
            if memories:
                import random
                random_memory = random.choice(memories)
                
                st.info("**Random Memory:**")
                st.write(random_memory.get('content', ''))
                st.json(random_memory.get('metadata', {}))
        except Exception as e:
            st.error(f"Error getting random memory: {e}")
    
    # Memory statistics
    st.subheader("📊 Live Statistics")
    
    try:
        memories = memory.list_memories(limit=1000)
        
        if memories:
            # Real-time metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Memories", len(memories))
            
            with col2:
                categories = set()
                for mem in memories:
                    category = mem.get('metadata', {}).get('category', 'uncategorized')
                    categories.add(category)
                st.metric("Categories", len(categories))
            
            with col3:
                total_length = sum(len(mem.get('content', '')) for mem in memories)
                st.metric("Total Content", f"{total_length:,} chars")
        else:
            st.info("No memories to display statistics for.")
    
    except Exception as e:
        st.error(f"Error loading statistics: {e}")


def show_settings(memory):
    """Show settings and configuration."""
    st.header("⚙️ Settings")
    
    st.subheader("Current Configuration")
    
    # Display current settings
    st.json({
        "vector_store": {
            "type": "faiss",
            "index_path": "./streamlit_index"
        },
        "embedder": {
            "type": "openai" if os.getenv('OPENAI_API_KEY') else "sentence_transformers",
            "model": "text-embedding-ada-002" if os.getenv('OPENAI_API_KEY') else "all-MiniLM-L6-v2"
        },
        "metadata_store": {
            "type": "sqlite",
            "database_path": "./streamlit_metadata.db"
        },
        "settings": {
            "max_memories": 10000,
            "similarity_threshold": 0.6,
            "default_top_k": 10
        }
    })
    
    st.subheader("Environment Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**API Keys:**")
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            st.success("✅ OpenAI API key found")
        else:
            st.warning("⚠️ OpenAI API key not found")
    
    with col2:
        st.write("**System Info:**")
        st.write(f"Python: {sys.version}")
        st.write(f"Platform: {sys.platform}")
    
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Memories", type="secondary"):
            try:
                memories = memory.list_memories(limit=10000)
                for mem in memories:
                    memory.delete(mem['memory_id'])
                st.success("✅ All memories cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing memories: {e}")
    
    with col2:
        if st.button("📊 Export Memories", type="secondary"):
            try:
                memories = memory.list_memories(limit=10000)
                if memories:
                    # Convert to JSON
                    export_data = json.dumps(memories, indent=2, default=str)
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download JSON",
                        data=export_data,
                        file_name=f"memorix_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No memories to export.")
            except Exception as e:
                st.error(f"❌ Error exporting memories: {e}")


if __name__ == "__main__":
    main() 