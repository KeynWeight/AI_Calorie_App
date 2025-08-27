# streamlit_components/sidebar_config.py
import streamlit as st
import os

def render_sidebar_config():
    """Render the sidebar configuration panel."""
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Configuration section
        render_api_config()
        
        # Model Configuration section
        render_model_config()
        
        # Analysis Settings section
        render_analysis_config()
        
        # System Status section
        render_system_status()
        
        # Advanced Settings toggle
        if st.checkbox("🔧 Show Advanced Settings"):
            st.session_state.show_advanced = True
            render_advanced_settings()
        else:
            st.session_state.show_advanced = False

def render_api_config():
    """Render API configuration section."""
    with st.expander("🔑 API Configuration", expanded=False):
        st.write("**Required API Keys:**")
        
        # OpenRouter API Key
        current_openrouter = st.session_state.get('openrouter_key', '')
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            value=current_openrouter,
            type="password",
            help="Required for vision and language models. Get it from https://openrouter.ai/"
        )
        
        if openrouter_key != current_openrouter:
            st.session_state.openrouter_key = openrouter_key
            if openrouter_key:
                st.success("✅ OpenRouter API key updated")
        
        # USDA API Key
        current_usda = st.session_state.get('usda_key', '')
        usda_key = st.text_input(
            "USDA API Key (Optional)",
            value=current_usda,
            type="password",
            help="For USDA food database enhancement. Get it from https://fdc.nal.usda.gov/api-key-signup.html"
        )
        
        if usda_key != current_usda:
            st.session_state.usda_key = usda_key
            if usda_key:
                st.success("✅ USDA API key updated")
        
        # Key status
        st.write("**Key Status:**")
        openrouter_status = "✅ Set" if openrouter_key else "❌ Missing"
        usda_status = "✅ Set" if usda_key else "⚠️ Optional"
        
        st.write(f"• OpenRouter: {openrouter_status}")
        st.write(f"• USDA: {usda_status}")
        
        # Save to environment option
        if st.button("💾 Save Keys to Environment"):
            if openrouter_key or usda_key:
                save_keys_to_env(openrouter_key, usda_key)
                st.success("Keys saved to environment!")
            else:
                st.warning("No keys to save")

def render_model_config():
    """Render model configuration section."""
    with st.expander("🤖 Model Configuration", expanded=False):
        st.write("**Vision Model:**")
        
        vision_models = [
            "qwen/qwen2.5-vl-72b-instruct:free",
            "qwen/qwen2-vl-7b-instruct:free", 
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "google/learnlm-1.5-pro-experimental:free"
        ]
        
        current_vision = st.session_state.get('vision_model', vision_models[0])
        vision_model = st.selectbox(
            "Vision Model",
            vision_models,
            index=vision_models.index(current_vision) if current_vision in vision_models else 0,
            help="Model for analyzing food images"
        )
        
        if vision_model != current_vision:
            st.session_state.vision_model = vision_model
        
        st.write("**Language Model:**")
        
        llm_models = [
            "meta-llama/llama-3.2-3b-instruct:free",
            "meta-llama/llama-3.2-1b-instruct:free",
            "qwen/qwen2.5-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free"
        ]
        
        current_llm = st.session_state.get('llm_model', llm_models[0])
        llm_model = st.selectbox(
            "Language Model",
            llm_models,
            index=llm_models.index(current_llm) if current_llm in llm_models else 0,
            help="Model for text processing and response generation"
        )
        
        if llm_model != current_llm:
            st.session_state.llm_model = llm_model
        
        # Model info
        st.info("💡 All models shown are free tier options from OpenRouter")

def render_analysis_config():
    """Render analysis configuration section."""
    with st.expander("🎛️ Analysis Settings", expanded=False):
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('confidence_threshold', 0.7),
            step=0.1,
            help="Minimum confidence score for accepting analysis results"
        )
        
        st.session_state.confidence_threshold = confidence_threshold
        
        # Caching settings
        enable_caching = st.checkbox(
            "Enable Response Caching",
            value=st.session_state.get('enable_caching', True),
            help="Cache API responses to avoid redundant calls for the same images"
        )
        
        st.session_state.enable_caching = enable_caching
        
        # Debug mode
        debug_mode = st.checkbox(
            "Debug Mode",
            value=st.session_state.get('debug_mode', False),
            help="Show detailed debug information and logs"
        )
        
        st.session_state.debug_mode = debug_mode
        
        # Cache management
        if enable_caching:
            st.write("**Cache Management:**")
            
            if st.button("🗑️ Clear Cache"):
                clear_cache()
                st.success("Cache cleared!")

def render_system_status():
    """Render system status section."""
    with st.expander("📊 System Status", expanded=True):
        # API Status
        openrouter_key = st.session_state.get('openrouter_key', '') or os.getenv('OPENROUTER_API_KEY', '')
        usda_key = st.session_state.get('usda_key', '') or os.getenv('USDA_API_KEY', '')
        
        if openrouter_key:
            st.success("✅ OpenRouter: Connected")
        else:
            st.error("❌ OpenRouter: API key required")
        
        # USDA Status
        if usda_key:
            # Check if MCP server is available
            mcp_server_exists = os.path.exists("food-data-central-mcp-server/dist/index.js")
            if mcp_server_exists:
                st.success("✅ USDA MCP: Available")
                st.session_state.mcp_available = True
            else:
                st.warning("⚠️ USDA MCP: Server not built")
                st.session_state.mcp_available = False
        else:
            st.info("ℹ️ USDA MCP: Optional (key not set)")
            st.session_state.mcp_available = False
        
        # Workflow Status
        if 'workflow_obj' in st.session_state and st.session_state.workflow_obj:
            st.success("✅ Workflow: Initialized")
        else:
            st.warning("⚠️ Workflow: Not initialized")
        
        # Current analysis status
        stage = st.session_state.get('analysis_stage', 'upload')
        stage_emojis = {
            'upload': '📤',
            'analyzing': '⏳',
            'validating': '✋',
            'modifying': '✏️',
            'usda_offer': '💭',
            'enhancing': '✨',
            'complete': '✅'
        }
        
        emoji = stage_emojis.get(stage, '❓')
        st.info(f"{emoji} Current Stage: {stage.title()}")
        
        # Cache statistics
        cache_info = get_cache_info()
        st.write(f"**Cache:** {cache_info['size']} items")

def render_advanced_settings():
    """Render advanced settings section."""
    st.markdown("### 🔧 Advanced Settings")
    
    # Performance settings
    with st.expander("⚡ Performance Settings"):
        max_retries = st.number_input(
            "Max API Retries",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of retry attempts for failed API calls"
        )
        
        timeout_seconds = st.number_input(
            "Request Timeout (seconds)",
            min_value=30,
            max_value=180,
            value=60,
            help="Timeout for API requests"
        )
    
    # Logging settings
    with st.expander("📝 Logging Configuration"):
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        log_level = st.selectbox(
            "Log Level",
            log_levels,
            index=1,  # Default to INFO
            help="Minimum log level to display"
        )
        
        log_to_file = st.checkbox(
            "Save Logs to File",
            value=True,
            help="Save detailed logs to file for debugging"
        )
    
    # Cache settings
    with st.expander("🗂️ Cache Configuration"):
        cache_size_mb = st.slider(
            "Max Cache Size (MB)",
            min_value=10,
            max_value=500,
            value=100,
            help="Maximum size of the response cache"
        )
        
        cache_ttl_hours = st.number_input(
            "Cache TTL (hours)",
            min_value=1,
            max_value=168,  # 1 week
            value=24,
            help="How long to keep cached responses"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View Cache Stats"):
                show_cache_stats()
        
        with col2:
            if st.button("🧹 Clear All Caches"):
                clear_all_caches()
                st.success("All caches cleared!")
    
    # Development settings
    with st.expander("🛠️ Development Settings"):
        dev_mode = st.checkbox(
            "Development Mode",
            value=False,
            help="Enable development features and verbose logging"
        )
        
        mock_api_calls = st.checkbox(
            "Mock API Calls",
            value=False,
            help="Use mock responses instead of real API calls (for testing)"
        )
        
        if st.button("🔄 Reset All Settings"):
            reset_all_settings()
            st.success("All settings reset to defaults!")

def save_keys_to_env(openrouter_key: str, usda_key: str):
    """Save API keys to .env file."""
    env_lines = []
    
    # Read existing .env file if it exists
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            env_lines = f.readlines()
    
    # Update or add keys
    def update_env_line(lines, key, value):
        key_line = f"{key}={value}\n"
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = key_line
                return
        # If not found, add it
        lines.append(key_line)
    
    if openrouter_key:
        update_env_line(env_lines, "OPENROUTER_API_KEY", openrouter_key)
        update_env_line(env_lines, "OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
    
    if usda_key:
        update_env_line(env_lines, "USDA_API_KEY", usda_key)
    
    # Write back to file
    with open(env_path, 'w') as f:
        f.writelines(env_lines)

def clear_cache():
    """Clear the application cache."""
    # Clear VLM cache
    try:
        import shutil
        cache_dirs = ['.cache', '.llm_cache', 'temp_uploads']
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        st.error(f"Failed to clear cache: {str(e)}")

def get_cache_info():
    """Get cache information."""
    cache_size = 0
    cache_dirs = ['.cache', '.llm_cache']
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            for root, dirs, files in os.walk(cache_dir):
                cache_size += len(files)
    
    return {"size": cache_size}

def show_cache_stats():
    """Show detailed cache statistics."""
    st.write("**Cache Statistics:**")
    
    cache_dirs = {
        '.cache': 'VLM Response Cache',
        '.llm_cache': 'LLM Response Cache',
        'temp_uploads': 'Temporary Uploads'
    }
    
    for cache_dir, description in cache_dirs.items():
        if os.path.exists(cache_dir):
            file_count = sum(len(files) for _, _, files in os.walk(cache_dir))
            dir_size = sum(
                os.path.getsize(os.path.join(root, file))
                for root, _, files in os.walk(cache_dir)
                for file in files
            )
            st.write(f"• {description}: {file_count} files ({dir_size / 1024:.1f} KB)")
        else:
            st.write(f"• {description}: No cache directory")

def clear_all_caches():
    """Clear all application caches."""
    clear_cache()
    
    # Also clear session state cache-related items
    cache_keys = [key for key in st.session_state.keys() if 'cache' in key.lower()]
    for key in cache_keys:
        del st.session_state[key]

def reset_all_settings():
    """Reset all settings to defaults."""
    # Settings to reset
    settings_keys = [
        'vision_model', 'llm_model', 'openrouter_key', 'usda_key',
        'confidence_threshold', 'enable_caching', 'debug_mode',
        'show_advanced'
    ]
    
    # Default values
    defaults = {
        'vision_model': "qwen/qwen2.5-vl-72b-instruct:free",
        'llm_model': "meta-llama/llama-3.2-3b-instruct:free",
        'openrouter_key': '',
        'usda_key': '',
        'confidence_threshold': 0.7,
        'enable_caching': True,
        'debug_mode': False,
        'show_advanced': False
    }
    
    for key in settings_keys:
        if key in st.session_state:
            st.session_state[key] = defaults.get(key, None)

def check_setup_status():
    """Check if the application is properly set up."""
    issues = []
    
    # Check API keys
    openrouter_key = st.session_state.get('openrouter_key', '') or os.getenv('OPENROUTER_API_KEY', '')
    if not openrouter_key:
        issues.append("OpenRouter API key is required for the app to function")
    
    # Check if models are properly configured
    if not st.session_state.get('vision_model'):
        issues.append("Vision model not selected")
    
    if not st.session_state.get('llm_model'):
        issues.append("Language model not selected")
    
    return issues

def render_setup_warnings():
    """Render setup warnings if any issues are found."""
    issues = check_setup_status()
    
    if issues:
        st.error("⚠️ Setup Issues Found:")
        for issue in issues:
            st.write(f"• {issue}")
        
        st.write("Please configure the settings in the sidebar before using the app.")
        return False
    
    return True