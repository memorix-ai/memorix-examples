# Contributing to Memorix Examples

Thank you for your interest in contributing to the Memorix Examples repository! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions from the community! Here are the main ways you can help:

### 🎯 Types of Contributions

1. **New Demos**: Create demos for new use cases and applications
2. **Improve Existing Demos**: Enhance current examples with better features
3. **Bug Fixes**: Report and fix issues in existing demos
4. **Documentation**: Improve README, comments, and documentation
5. **Testing**: Add tests and improve test coverage
6. **Performance**: Optimize demos for better performance
7. **Examples**: Share how you're using Memorix in your projects

### 🚀 Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/memorix-ai/memorix-examples.git
   cd memorix-examples
   ```

2. **Set up your development environment**
   ```bash
   pip install -r requirements.txt
   pip install memorix-sdk
   ```

3. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**
   - Follow the coding standards below
   - Add tests if applicable
   - Update documentation

5. **Test your changes**
   ```bash
   # Run the demo you're working on
   python demos/your_demo.py --test-mode
   
   # Run all tests
   pytest tests/
   ```

6. **Submit a pull request**
   - Provide a clear description of your changes
   - Include any relevant issue numbers
   - Add screenshots for UI changes

## 📝 Coding Standards

### Python Code Style

- **Follow PEP 8**: Use consistent formatting
- **Type hints**: Add type hints for function parameters and return values
- **Docstrings**: Include comprehensive docstrings for all functions and classes
- **Comments**: Add inline comments for complex logic
- **Naming**: Use descriptive variable and function names

### Demo Structure

Each demo should follow this structure:

```python
#!/usr/bin/env python3
"""
Memorix SDK - Demo Name
=======================

Brief description of what this demo shows.

Features:
- Feature 1
- Feature 2
- Feature 3

Perfect for [use case]!
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add the parent directory to the path to import memorix
sys.path.append(str(Path(__file__).parent.parent.parent))

from memorix import MemoryAPI, Config


def setup_config():
    """Set up configuration for the demo."""
    # Load configuration from YAML file
    config_path = Path(__file__).parent.parent / "configs" / "example_config.yaml"
    
    if config_path.exists():
        config = Config(str(config_path))
    else:
        config = Config()
        # Set up default configuration
    
    return config


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Memorix SDK Demo Name')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode (no user interaction)')
    args = parser.parse_args()
    
    try:
        # Demo implementation
        pass
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1
    finally:
        # Cleanup
        pass
    
    return 0


if __name__ == "__main__":
    exit(main())
```

### Configuration Management

- **Use YAML configs**: All demos should use `configs/example_config.yaml` as a base
- **Environment variables**: Use environment variables for API keys
- **Fallback gracefully**: Provide sensible defaults when configs are missing

### Error Handling

- **Graceful degradation**: Handle missing API keys and services gracefully
- **Informative messages**: Provide clear error messages and suggestions
- **Test mode**: Support `--test-mode` flag for CI/CD testing

## 🧪 Testing Guidelines

### Demo Testing

- **Test mode**: All demos should support `--test-mode` for automated testing
- **No user interaction**: Test mode should run without requiring user input
- **Cleanup**: Automatically clean up test files in test mode
- **Error handling**: Test error conditions and edge cases

### Running Tests

```bash
# Test a specific demo
python demos/01_basic_usage.py --test-mode

# Test all demos
for demo in demos/*.py; do
    python "$demo" --test-mode
done
```

### CI/CD Integration

- **GitHub Actions**: All demos are tested in CI
- **Multiple Python versions**: Tests run on Python 3.8, 3.9, 3.10, 3.11
- **Dependency caching**: CI caches dependencies for faster builds
- **Artifact upload**: Test results are uploaded as artifacts

## 📚 Documentation Standards

### README Updates

When adding new demos:

1. **Update the demo list** in the main README
2. **Add a brief description** of what the demo shows
3. **Include usage examples** if applicable
4. **Update the "What You'll Learn" section**

### Code Documentation

- **Module docstrings**: Describe the purpose of each demo
- **Function docstrings**: Use Google-style docstrings
- **Inline comments**: Explain complex logic
- **Type hints**: Add comprehensive type annotations

### Example Documentation

```python
def demo_feature(memory: MemoryAPI) -> List[Dict[str, Any]]:
    """
    Demonstrate a specific feature of the Memorix SDK.
    
    Args:
        memory: The MemoryAPI instance to use for the demo
        
    Returns:
        List of results from the demo
        
    Raises:
        ValueError: If the demo cannot be completed
        
    Example:
        >>> results = demo_feature(memory)
        >>> print(f"Found {len(results)} results")
    """
    # Implementation here
    pass
```

## 🎨 Demo Categories

### Core Demos (MVP)
- **Basic Usage**: Essential features and concepts
- **Chatbot Memory**: Conversational AI with memory
- **Vector Store Comparison**: Performance and feature comparison

### Advanced Demos
- **Embedding Models**: Different embedding model comparisons
- **Knowledge Base**: Document processing and search
- **Custom Components**: Building custom vector stores and embedders

### Real-World Applications
- **Document Q&A**: Question answering over documents
- **Code Assistant**: AI coding assistant with memory
- **Research Assistant**: Research paper analysis
- **Customer Support**: Support chatbot with context

### Experimental Demos
- **Memory Compression**: Memory summarization techniques
- **Multi-modal Memory**: Text and image storage
- **Temporal Memory**: Time-aware retrieval
- **Collaborative Memory**: Shared memory across users

## 🔧 Development Setup

### Prerequisites

- Python 3.8+
- Git
- API keys for services (OpenAI, Google, etc.)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/memorix-ai/memorix-examples.git
   cd memorix-examples
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install memorix-sdk
   ```

4. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   ```

5. **Run a demo**
   ```bash
   python demos/01_basic_usage.py
   ```

### Development Tools

- **Code formatting**: Use `black` for consistent formatting
- **Linting**: Use `flake8` for code quality checks
- **Type checking**: Use `mypy` for static type analysis
- **Testing**: Use `pytest` for unit tests

```bash
# Format code
black demos/

# Lint code
flake8 demos/

# Type check
mypy demos/

# Run tests
pytest tests/
```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment details** (OS, Python version, dependencies)
5. **Error messages** and stack traces
6. **Screenshots** if applicable

### Feature Requests

When requesting features, please include:

1. **Clear description** of the feature
2. **Use case** and motivation
3. **Proposed implementation** if you have ideas
4. **Examples** of similar features in other projects

## 📋 Pull Request Guidelines

### Before Submitting

1. **Test your changes**: Run the demo in test mode
2. **Update documentation**: Update README and docstrings
3. **Check formatting**: Run `black` and `flake8`
4. **Add tests**: Include tests for new functionality
5. **Update requirements**: Add new dependencies to `requirements.txt`

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tested locally
- [ ] Added tests
- [ ] All tests pass

## Documentation
- [ ] Updated README
- [ ] Updated docstrings
- [ ] Added inline comments

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No breaking changes
- [ ] Dependencies updated
```

## 🏷️ Versioning

We follow [Semantic Versioning](https://semver.org/) for releases:

- **Major version**: Breaking changes
- **Minor version**: New features, backward compatible
- **Patch version**: Bug fixes, backward compatible

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.

## 🎉 Recognition

Contributors will be recognized in:

- **README**: List of contributors
- **Release notes**: Credit for significant contributions
- **Documentation**: Attribution for examples and tutorials

## 📞 Getting Help

If you need help with contributing:

- **GitHub Issues**: Open an issue for questions
- **GitHub Discussions**: Join community discussions
- **Documentation**: Check the main SDK documentation
- **Email**: Contact support@memorix.ai

## 🙏 Thank You

Thank you for contributing to the Memorix Examples repository! Your contributions help make the Memorix SDK more accessible and useful for the entire AI community.

---

**Happy coding! 🚀** 