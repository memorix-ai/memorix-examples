#!/usr/bin/env python3
"""
Memorix SDK Examples - Setup Script
===================================

This script helps you set up the Memorix SDK examples repository.
It will:
1. Install required dependencies
2. Create configuration files
3. Set up environment variables
4. Run initial tests
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔧 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install memorix-sdk in development mode if available
    memorix_path = Path("../memorix-sdk")
    if memorix_path.exists():
        print("\n🔧 Installing Memorix SDK in development mode...")
        if not run_command(f"pip install -e {memorix_path}", "Installing Memorix SDK"):
            print("⚠️  Could not install Memorix SDK in development mode. Using pip install.")
            if not run_command("pip install memorix-sdk", "Installing Memorix SDK from PyPI"):
                return False
    else:
        print("\n🔧 Installing Memorix SDK from PyPI...")
        if not run_command("pip install memorix-sdk", "Installing Memorix SDK"):
            return False
    
    return True


def create_config_files():
    """Create configuration files."""
    print("\n⚙️  Creating configuration files...")
    
    # Create configs directory if it doesn't exist
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Copy example config
    example_config = Path("configs/example_config.yaml")
    if example_config.exists():
        # Create user config
        user_config = Path("configs/my_config.yaml")
        if not user_config.exists():
            shutil.copy(example_config, user_config)
            print("✅ Created configs/my_config.yaml")
        else:
            print("ℹ️  configs/my_config.yaml already exists")
    else:
        print("⚠️  Example config not found")
    
    return True


def setup_environment():
    """Set up environment variables."""
    print("\n🌍 Setting up environment...")
    
    # Check for API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI API key for embeddings and completions",
        "GOOGLE_API_KEY": "Google API key for Gemini embeddings",
        "ANTHROPIC_API_KEY": "Anthropic API key for Claude models"
    }
    
    missing_keys = []
    for key, description in api_keys.items():
        if not os.getenv(key):
            missing_keys.append((key, description))
    
    if missing_keys:
        print("⚠️  The following API keys are not set:")
        for key, description in missing_keys:
            print(f"  - {key}: {description}")
        
        print("\n💡 To set API keys, run:")
        for key, _ in missing_keys:
            print(f"  export {key}='your-api-key-here'")
        
        print("\n💡 Or add them to your .env file:")
        env_file = Path(".env")
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write("# Memorix SDK Examples - Environment Variables\n")
                f.write("# Add your API keys here\n\n")
                for key, description in missing_keys:
                    f.write(f"# {description}\n")
                    f.write(f"# {key}=your-api-key-here\n\n")
            print("✅ Created .env file template")
    else:
        print("✅ All API keys are set!")
    
    return True


def run_tests():
    """Run initial tests."""
    print("\n🧪 Running initial tests...")
    
    # Test basic import
    try:
        import memorix
        print("✅ Memorix SDK import successful!")
    except ImportError as e:
        print(f"❌ Memorix SDK import failed: {e}")
        return False
    
    # Test basic demo
    demo_file = Path("demos/01_basic_usage.py")
    if demo_file.exists():
        print("🔧 Testing basic usage demo...")
        if run_command("python demos/01_basic_usage.py --test", "Running basic usage demo"):
            print("✅ Basic usage demo test passed!")
        else:
            print("⚠️  Basic usage demo test failed (this is normal if API keys are not set)")
    else:
        print("⚠️  Basic usage demo not found")
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "demos",
        "configs", 
        "web_demo",
        "streamlit_app",
        "gradio_app",
        "notebooks",
        "templates/chatbot",
        "templates/knowledge_base",
        "templates/document_qa",
        "templates/code_assistant",
        "benchmarks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True


def show_next_steps():
    """Show next steps for the user."""
    print("\n🎉 Setup completed successfully!")
    print("\n📚 Next Steps:")
    print("1. Set up your API keys (see .env file)")
    print("2. Run a demo: python demos/01_basic_usage.py")
    print("3. Explore the examples:")
    print("   - demos/02_chatbot_memory.py")
    print("   - demos/03_knowledge_base.py")
    print("   - demos/04_vector_store_comparison.py")
    print("   - demos/05_embedding_models.py")
    print("4. Try the web interface: streamlit run streamlit_app/app.py")
    print("5. Use a template: python templates/chatbot/chatbot_template.py")
    
    print("\n📖 Documentation:")
    print("- README.md: Overview and getting started")
    print("- demos/: Interactive examples")
    print("- templates/: Ready-to-use templates")
    print("- configs/: Configuration examples")
    
    print("\n🤝 Community:")
    print("- GitHub: https://github.com/your-org/memorix-sdk")
    print("- Issues: https://github.com/your-org/memorix-sdk/issues")
    print("- Discussions: https://github.com/your-org/memorix-sdk/discussions")


def main():
    """Main setup function."""
    print("🚀 Memorix SDK Examples - Setup")
    print("=" * 50)
    print("This script will help you set up the Memorix SDK examples repository.")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Create config files
    if not create_config_files():
        return 1
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but setup can continue...")
    
    # Show next steps
    show_next_steps()
    
    return 0


if __name__ == "__main__":
    exit(main()) 