# Memorix SDK Examples & Demos 🧠

Welcome to the Memorix SDK Examples repository! This collection of demos showcases the power and flexibility of the Memorix memory management system for AI applications.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/memorix-examples.git
cd memorix-examples

# Install dependencies
pip install -r requirements.txt

# Set up your API keys
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"

# Run any demo
python demos/01_basic_usage.py
```

## 📚 Demo Collection

### 🎯 Core Demos
- **[01_basic_usage.py](demos/01_basic_usage.py)** - Get started with Memorix in 5 minutes
- **[02_chatbot_memory.py](demos/02_chatbot_memory.py)** - Build a conversational AI with memory
- **[03_knowledge_base.py](demos/03_knowledge_base.py)** - Create a searchable knowledge base
- **[04_vector_store_comparison.py](demos/04_vector_store_comparison.py)** - Compare different vector stores

### 🔧 Advanced Demos
- **[05_embedding_models.py](demos/05_embedding_models.py)** - Explore different embedding models
- **[06_metadata_management.py](demos/06_metadata_management.py)** - Advanced metadata operations
- **[07_batch_operations.py](demos/07_batch_operations.py)** - Efficient bulk memory operations
- **[08_custom_components.py](demos/08_custom_components.py)** - Build custom vector stores and embedders

### 🎨 Real-World Applications
- **[09_document_qa.py](demos/09_document_qa.py)** - Question answering over documents
- **[10_code_assistant.py](demos/10_code_assistant.py)** - AI coding assistant with memory
- **[11_research_assistant.py](demos/11_research_assistant.py)** - Research paper analysis and retrieval
- **[12_customer_support.py](demos/12_customer_support.py)** - Customer support chatbot with context

### 🧪 Experimental Demos
- **[13_memory_compression.py](demos/13_memory_compression.py)** - Memory summarization techniques
- **[14_multi_modal_memory.py](demos/14_multi_modal_memory.py)** - Text and image memory storage
- **[15_temporal_memory.py](demos/15_temporal_memory.py)** - Time-aware memory retrieval
- **[16_collaborative_memory.py](demos/16_collaborative_memory.py)** - Shared memory across users

## 🎬 Interactive Demos

### Web Applications
- **[web_demo/](web_demo/)** - Interactive web interface for memory management
- **[streamlit_app/](streamlit_app/)** - Streamlit-based memory explorer
- **[gradio_app/](gradio_app/)** - Gradio interface for quick prototyping

### Jupyter Notebooks
- **[notebooks/](notebooks/)** - Jupyter notebooks for interactive learning
  - `01_getting_started.ipynb` - Interactive tutorial
  - `02_advanced_features.ipynb` - Advanced concepts
  - `03_real_world_applications.ipynb` - Practical examples

## 🏗️ Project Templates

### Starter Templates
- **[templates/chatbot/](templates/chatbot/)** - Ready-to-use chatbot template
- **[templates/knowledge_base/](templates/knowledge_base/)** - Knowledge base starter
- **[templates/document_qa/](templates/document_qa/)** - Document Q&A system
- **[templates/code_assistant/](templates/code_assistant/)** - Code assistant template

## 🎯 What You'll Learn

### Core Concepts
- ✅ Memory storage and retrieval
- ✅ Vector similarity search
- ✅ Metadata management
- ✅ Configuration management
- ✅ Different vector stores (FAISS, Qdrant)
- ✅ Various embedding models (OpenAI, Gemini, Sentence Transformers)

### Advanced Features
- ✅ Batch operations
- ✅ Custom components
- ✅ Memory compression
- ✅ Multi-modal storage
- ✅ Temporal memory
- ✅ Collaborative memory

### Real-World Applications
- ✅ Building chatbots with memory
- ✅ Creating knowledge bases
- ✅ Document question answering
- ✅ Code assistance
- ✅ Research paper analysis
- ✅ Customer support systems

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- API keys for your chosen services (OpenAI, Google, etc.)

### Installation
```bash
pip install memorix-sdk
pip install -r requirements.txt
```

### Configuration
Copy the example configuration:
```bash
cp configs/example_config.yaml my_config.yaml
# Edit my_config.yaml with your API keys and preferences
```

## 🎨 Demo Highlights

### 🤖 Conversational AI with Memory
Build chatbots that remember conversations and context:
```python
from memorix import MemoryAPI, Config

# Initialize memory system
memory = MemoryAPI(Config('config.yaml'))

# Store conversation context
memory.store("User asked about Python programming", metadata={"session": "user_123"})

# Retrieve relevant context for responses
context = memory.retrieve("Python programming", top_k=3)
```

### 📚 Knowledge Base Creation
Create searchable knowledge bases from documents:
```python
# Store document chunks with metadata
for chunk in document_chunks:
    memory.store(
        chunk.text,
        metadata={
            "source": "document.pdf",
            "page": chunk.page,
            "section": chunk.section
        }
    )

# Search across all documents
results = memory.retrieve("machine learning algorithms", top_k=10)
```

### 🔍 Advanced Search
Leverage metadata for powerful filtering:
```python
# Search with metadata filters
results = memory.retrieve(
    "neural networks",
    metadata_filter={"category": "AI", "difficulty": "advanced"}
)
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Add New Demos**: Create demos for new use cases
2. **Improve Existing Demos**: Enhance current examples
3. **Fix Issues**: Report and fix bugs
4. **Documentation**: Improve docs and comments
5. **Share Use Cases**: Tell us how you're using Memorix

### Contributing Guidelines
- Follow the existing code style
- Add comprehensive comments
- Include requirements in `requirements.txt`
- Test your demos before submitting
- Update this README if adding new demos

## 📊 Performance Benchmarks

See [benchmarks/](benchmarks/) for performance comparisons:
- Vector store performance
- Embedding model speed
- Memory retrieval accuracy
- Scalability tests

## 🎯 Use Cases

### For Developers
- Build AI applications with persistent memory
- Create intelligent chatbots
- Implement document search systems
- Build code assistance tools

### For Researchers
- Experiment with different embedding models
- Test vector store performance
- Research memory compression techniques
- Explore multi-modal memory systems

### For Businesses
- Customer support automation
- Knowledge management systems
- Document processing pipelines
- AI-powered search engines

## 📈 Community Impact

Join our growing community: Memorix

## 📞 Support & Community

- 📖 **Documentation**: [docs.memorix.ai](https://docs.memorix.ai)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/memorix-sdk/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/memorix-sdk/issues)
- 📧 **Email**: support@memorix.ai
- 🐦 **Twitter**: [@MemorixAI](https://twitter.com/MemorixAI)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Ready to build the future of AI memory? Start with these demos! 🚀** 