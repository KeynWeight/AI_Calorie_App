# 🍽️ Calorie App - AI-Powered Nutrition Analyzer

An intelligent nutrition analysis system that leverages computer vision and large language models to analyze food images and provide comprehensive nutritional information with human-in-the-loop validation.

## ✨ Key Features

- **🤖 AI-Powered Analysis**: Advanced vision models identify dishes and ingredients
- **🔢 Precise Nutrition Calculation**: Detailed estimates of calories, macronutrients, and micronutrients
- **✋ Human-in-the-Loop Workflow**: Review, validate, and modify AI results for accuracy
- **🏛️ USDA Integration**: Enhanced nutrition data from official USDA FoodData Central database
- **📊 Multiple Output Formats**: Natural language summaries, JSON data, and interactive visualizations
- **🌐 Web Interface**: Beautiful Streamlit-powered web application
- **⚙️ Configurable**: Support for multiple AI models and customizable settings

## 📁 Project Structure

```
calorie_app/
├── 📁 src/calorie_app/              # Main application package
│   ├── models/                       # Pydantic data models for type safety
│   ├── services/                     # Core business logic (AI, vision, nutrition)
│   ├── tools/                        # Specialized utility tools
│   ├── utils/                        # Shared helper functions
│   └── workflows/                    # Multi-step workflow orchestration
├── 📁 streamlit_components/          # Modular Streamlit UI components
├── 📁 tests/                         # Comprehensive test suite
├── 📁 docs/                          # Project documentation
├── 📁 assets/                        # Static assets and generated files
├── 📁 logs/                          # Application logs
├── 📁 food-data-central-mcp-server/  # USDA MCP server integration
├── 📄 app.py                         # Main Streamlit web application
├── 📄 main.py                        # CLI/programmatic usage examples
└── 📄 pyproject.toml                 # Python dependencies and project config
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your OpenRouter API key (required) and USDA API key (optional)
```

### 2. Install Dependencies

```bash
# Using UV (recommended - faster and more reliable)
uv sync

# Or using pip
pip install -e .
```

### 3. Optional: Setup USDA Enhancement

For enhanced nutrition data from USDA FoodData Central:

```bash
cd food-data-central-mcp-server
npm install && npm run build
cd ..
```

### 4. Launch the Application

#### 🌐 Web Interface (Recommended)
```bash
uv run streamlit run app.py
# Opens at http://localhost:8501
```

#### 💻 Programmatic Usage
```bash
uv run python main.py
```

## 💡 Usage Guide

### 🌐 Web Interface (Streamlit App)

The intuitive web interface guides you through a 5-step process:

1. **📤 Upload Image**: Multiple options available
   - Drag & drop food images
   - Browse and select files
   - Take photos with camera (mobile-friendly)
   - Try sample images

2. **🤖 AI Analysis**: Advanced computer vision processing
   - Automatic dish and ingredient identification
   - Portion size estimation
   - Initial nutrition calculation
   - Real-time progress tracking

3. **✅ Review & Validate**: Human-in-the-loop quality control
   - Review AI-identified dishes and ingredients
   - Check confidence scores and accuracy
   - Approve results or proceed to modifications

4. **✏️ Modify (Optional)**: Fine-tune the analysis
   - Adjust ingredient weights with intuitive sliders
   - Add missing ingredients
   - Remove incorrect identifications
   - Edit dish names and descriptions

5. **✨ USDA Enhancement (Optional)**: Professional-grade data
   - Leverage official USDA FoodData Central database
   - AI agent automatically matches ingredients
   - Enhanced micronutrient profiles

6. **📊 Results Dashboard**: Comprehensive nutrition insights
   - Interactive charts and visualizations
   - Detailed macro/micronutrient breakdown
   - Export options (PDF, CSV, JSON)

### 💻 Programmatic API

For developers and advanced users:

```python
from calorie_app.workflows.nutrition_workflow import NutritionWorkflow

# Initialize with your preferred models
workflow = NutritionWorkflow(
    vision_model="qwen/qwen2.5-vl-72b-instruct:free",
    vision_api_key="your_openrouter_key",
    llm_model="meta-llama/llama-3.2-3b-instruct:free",
    llm_api_key="your_openrouter_key"
)

# Analyze a food image
results = workflow.start_analysis("path/to/food/image.jpg")

# Process human validation and get final results
final_results = workflow.submit_human_validation(
    thread_id="analysis_thread",
    approved=True,
    modifications=None,
    wants_usda_info=True
)

print(f"Total calories: {final_results.total_nutrition.calories}")
print(f"Protein: {final_results.total_nutrition.protein}g")
```

## ⚙️ Configuration

### 🔑 Required API Keys

| Service | Purpose | Required | Get Key |
|---------|---------|----------|----------|
| **OpenRouter** | Vision & language models | ✅ Yes | [Get API Key](https://openrouter.ai/) |
| **USDA FoodData Central** | Enhanced nutrition data | 🔶 Optional | [Get API Key](https://fdc.nal.usda.gov/api-key-signup.html) |

### 🌐 Environment Variables

Create a `.env` file in the project root:

```bash
# Required - OpenRouter API configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_API_URL=https://openrouter.ai/api/v1

# Optional - USDA enhancement
USDA_API_KEY=your_usda_api_key_here

# Optional - Application settings
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

### 🤖 Supported AI Models

**Vision Models** (for image analysis):
- `qwen/qwen2.5-vl-72b-instruct:free` (recommended)
- `meta-llama/llama-3.2-11b-vision-instruct:free`
- `google/learnlm-1.5-pro-experimental:free`

**Language Models** (for text processing):
- `meta-llama/llama-3.2-3b-instruct:free` (recommended)
- `microsoft/phi-3.5-mini-instruct:free`
- `google/gemma-2-9b-it:free`

*All models use free tiers from OpenRouter - no additional costs!*

## 👨‍💻 Development

### 🧪 Running Tests

```bash
# Run complete test suite
uv run pytest

# Run with Python script (alternative method)
uv run python run_tests.py

# Generate coverage report
uv run pytest --cov=calorie_app --cov-report=html

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m e2e           # End-to-end tests
uv run pytest -v               # Verbose output
```

### ✨ Code Quality & Formatting

```bash
# Auto-format code
uv run black src/ tests/ streamlit_components/
uv run isort src/ tests/ streamlit_components/

# Lint and check code quality
uv run ruff check src/ tests/ streamlit_components/
uv run ruff check --fix  # Auto-fix issues

# Type checking
uv run mypy src/
```

### 🛠️ Development Tools

```bash
# Start development server with auto-reload
uv run streamlit run app.py --server.runOnSave=true

# Run with debug logging
LOG_LEVEL=DEBUG uv run streamlit run app.py

# Profile performance
uv run python -m cProfile main.py
```

## 🏢 Architecture Overview

The application follows **clean architecture principles** with clear separation of concerns:

### 📊 Core Layers

- **📊 Models Layer**: Pydantic data models ensuring type safety and validation
- **🔧 Services Layer**: Core business logic for AI vision, LLM processing, and nutrition analysis
- **🔄 Workflows Layer**: Orchestrates complex multi-step processes with state management
- **🤖 Tools Layer**: Specialized utilities for specific tasks (image processing, data export)
- **⚙️ Utils Layer**: Shared utilities (caching, logging, validation, error handling)
- **🌐 Components Layer**: Modular Streamlit UI components for the web interface

### 📊 Data Flow

```
📤 Image Upload → 🤖 AI Analysis → ✅ Human Review → ✏️ Modify → ✨ USDA Enhancement → 📊 Results
```

### 🔌 Integration Points

- **OpenRouter API**: Multi-model AI inference
- **USDA FoodData Central**: Official nutrition database via MCP server
- **Streamlit Framework**: Interactive web interface
- **File System**: Image uploads and result caching

## 🚀 Performance & Scalability

### 💾 Caching Strategy
- **Response Caching**: Avoid redundant API calls
- **Image Processing Cache**: Speed up repeated analyses
- **Model Result Cache**: Persistent storage of AI outputs

### 📦 Resource Management
- **Lazy Loading**: Components loaded on demand
- **Memory Optimization**: Efficient image processing
- **Background Processing**: Non-blocking operations

## 🔒 Security & Privacy

- **API Key Protection**: Secure environment variable handling
- **Image Privacy**: Temporary uploads with automatic cleanup
- **No Data Persistence**: Images and results not stored permanently
- **Rate Limiting**: Built-in OpenRouter API throttling

## 🌍 Deployment Options

### ☁️ Cloud Platforms
- **Streamlit Community Cloud**: Free hosting with GitHub integration
- **Heroku**: Container-based deployment
- **AWS/GCP/Azure**: Full cloud infrastructure
- **Railway/Render**: Modern platform-as-a-service

### 📦 Docker Deployment
```bash
# Build container
docker build -t calorie-app .

# Run with environment variables
docker run -p 8501:8501 -e OPENROUTER_API_KEY=your_key -e USDA_API_KEY=your_usda_key calorie-app

# Access at http://localhost:8501
```

## 🎆 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 📈 Development Workflow
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

## 🕰️ Troubleshooting

### Common Issues

**"ModuleNotFoundError" when running**
```bash
# Ensure you're in the right directory and dependencies are installed
pwd  # Should show /path/to/calorie_app
uv sync  # Reinstall dependencies
```

**"API key invalid" errors**
```bash
# Check your .env file exists and has correct keys
cat .env
# Verify API key format and permissions at OpenRouter
```

**Streamlit app won't start**
```bash
# Try with explicit Python path
uv run python -m streamlit run app.py
# Check for port conflicts
lsof -i :8501
```

**USDA enhancement not working**
```bash
# Rebuild the MCP server
cd food-data-central-mcp-server
npm run build
cd ..
```


## 🙏 Acknowledgments

- **OpenRouter** for providing free access to multiple AI models
- **USDA FoodData Central** for comprehensive nutrition data
- **Streamlit** for the excellent web app framework
- **The open-source community** for the amazing tools and libraries

---

<div align="center">

**🍽️ Ready to analyze your food? Get started now! ✨**

[View Demo](https://ai-calorie-app-agvf.onrender.com/)

</div>
