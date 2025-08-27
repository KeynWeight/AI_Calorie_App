# Project Structure After Cleanup

## 📂 Directory Layout

```
calorie_app/
├── 📁 src/calorie_app/          # Main Python package
│   ├── models/                  # Data models (Pydantic)
│   ├── services/                # Core business logic
│   ├── tools/                   # Utility tools
│   ├── utils/                   # Helper functions
│   └── workflows/               # Main workflow orchestration
├── 📁 streamlit_components/     # Streamlit UI components
├── 📁 tests/                    # Comprehensive test suite
├── 📁 logs/                     # Application logs
├── 📁 assets/                   # Static assets and sample images
├── 📁 food-data-central-mcp-server/  # USDA MCP server
├── 📁 temp_uploads/             # Temporary uploaded images
├── 📁 htmlcov/                  # Test coverage reports
├── 📄 app.py                    # Main Streamlit application
├── 📄 main.py                   # CLI example usage
├── 📄 run_tests.py              # Test runner script
├── 📄 pyproject.toml            # Python dependencies and config
├── 📄 uv.lock                   # UV dependency lock file
├── 📄 Dockerfile                # Container deployment config
├── 📄 README.md                 # Main project documentation
├── 📄 README_STREAMLIT.md       # Streamlit app guide
├── 📄 PROJECT_STRUCTURE.md      # This file - project organization
└── 📄 .env.example              # Environment variables template
```

## 🚀 Entry Points

### Web Application
```bash
# Start the Streamlit web app
uv run streamlit run app.py
# Opens at http://localhost:8501
```

### CLI Usage
```bash
# Run the programmatic example
uv run python main.py
```

## 📊 Current Directory Structure

### Core Application Components
- **src/calorie_app/**: Main Python package with modular architecture
- **streamlit_components/**: UI components for the web interface
- **tests/**: Comprehensive test suite with multiple test categories
- **logs/**: Application logging with detailed request/response tracking

### Assets & Data
- **assets/food_image/**: Sample food images (hainan.jpg, pasta.jpg)
- **temp_uploads/**: Temporary storage for uploaded images
- **htmlcov/**: Test coverage HTML reports

### Configuration & Deployment
- **food-data-central-mcp-server/**: TypeScript MCP server for USDA integration
- **Dockerfile**: Container deployment configuration
- **pyproject.toml**: Python dependencies, dev tools, and project metadata
- **uv.lock**: UV package manager lock file

## ✅ Project Status

- **✅ Application Structure**: Properly organized
- **✅ Dependencies**: pyproject.toml configured
- **✅ Import Tests**: All modules import successfully
- **✅ Streamlit App**: Starts and runs properly
- **✅ CLI Example**: main.py imports and initializes correctly

## ⚙️ Configuration Files

### 📦 Python Configuration (`pyproject.toml`)

**Dependencies Management:**
```toml
[project]
name = "calorie-app"
version = "0.1.0"
description = "AI-powered nutrition analysis system"
dependencies = [
    "aiofiles>=23.0.0",       # Async file operations
    "langchain>=0.3.27",      # LLM framework
    "langchain-openai>=0.3.30", # OpenAI integration
    "langchain-mcp-adapters>=0.1.0", # MCP protocol
    "opencv-python>=4.12.0",  # Computer vision
    "pillow>=10.0.0",         # Image processing
    "polars>=1.32.3",         # Fast DataFrame library
    "pydantic>=2.11.7",       # Data validation
    "streamlit>=1.28.0",      # Web framework
    "plotly>=5.15.0",         # Interactive charts
    "pandas>=2.0.0",          # Data manipulation
    "reportlab>=4.0.0",       # PDF generation
    "scikit-learn>=1.7.1",    # Machine learning
    "ultralytics>=8.3.180",   # YOLO models
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",           # Linting and formatting
    "black>=23.0.0",         # Code formatting
    "isort>=5.12.0",         # Import sorting
    "mypy>=1.0.0",           # Type checking
    "pytest>=8.4.1",         # Testing framework
    "pytest-cov>=4.0.0",     # Coverage reporting
    "pytest-asyncio>=1.1.0", # Async testing
    "safety>=2.3.0",         # Security scanning
    "bandit>=1.7.5",         # Security linting
]
```

### ⚙️ Environment Configuration (`.env.example`)

**Required Settings:**
```bash
# Core API Keys
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_API_URL=https://openrouter.ai/api/v1

# Optional USDA Enhancement
USDA_API_KEY=your_usda_key_here

# Application Settings
LOG_LEVEL=INFO
CACHE_ENABLED=true
MAX_UPLOAD_SIZE=10

# Development Settings
DEBUG_MODE=false
STREAMLIT_SERVER_PORT=8501
```

## 🚀 Getting Started Guide

### 🔍 Prerequisites Checklist

- [ ] **Python 3.11+** installed (3.13+ recommended)
- [ ] **Node.js 18+** (for USDA server)
- [ ] **Git** for version control
- [ ] **OpenRouter API Key** ([get free key](https://openrouter.ai/))
- [ ] **USDA API Key** (optional - [get free key](https://fdc.nal.usda.gov/api-key-signup.html))

### 🔧 Setup Process

```bash
# 1. Clone and navigate
git clone <repository-url>
cd calorie_app

# 2. Create environment and install dependencies
uv sync                    # Using UV (recommended)
# OR
pip install -e .          # Using pip

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Optional: Setup USDA enhancement
cd food-data-central-mcp-server
npm install && npm run build
cd ..

# 5. Launch application
uv run streamlit run app.py
```

### ✅ Project Health Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Core Package** | ✅ Ready | All modules import successfully |
| **Streamlit App** | ✅ Ready | Web interface fully functional |
| **CLI Interface** | ✅ Ready | Programmatic access working |
| **USDA MCP Server** | 🔶 Optional | TypeScript server, requires npm build |
| **Test Suite** | ✅ Ready | 12+ test markers, comprehensive coverage |
| **Documentation** | ✅ Complete | 3 README files, all guides updated |
| **Docker Support** | ✅ Ready | Dockerfile included for deployment |
| **UV Support** | ✅ Ready | Modern Python package management |

### 🔗 Quick Links

- **📚 [Main README](README.md)** - Project overview and features
- **🌐 [Streamlit Guide](README_STREAMLIT.md)** - Web interface documentation
- **🧪 [Testing Guide](tests/README.md)** - Running tests and development
- **🏛️ [USDA Server](food-data-central-mcp-server/README.md)** - Database integration setup

---

## 🎉 Project Status: Ready to Use!

The AI nutrition analyzer is fully organized, documented, and ready for:

- **👥 End Users**: Launch the Streamlit app and start analyzing food
- **👨‍💻 Developers**: Integrate via the programmatic API
- **🔬 Researchers**: Extend functionality and contribute improvements
- **🏢 Organizations**: Deploy to production environments

**Ready to start? Run `uv run streamlit run app.py` and begin analyzing your food! 🍽️✨**
