# streamlit_components/session_manager.py
import streamlit as st
import uuid

def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        'analysis_stage': 'upload',  # upload, analyzing, validating, modifying, usda_offer, enhancing, complete
        'thread_id': str(uuid.uuid4()),
        'uploaded_image_path': None,
        'uploaded_image_name': None,
        'original_analysis': None,
        'final_analysis': None,
        'enhanced_analysis': None,
        'validation_result': None,
        'validation_prompt': None,
        'natural_response': None,
        'workflow_obj': None,
        'mcp_available': False,
        'wants_usda_info': False,
        'usda_matches': 0,
        'modifications': {},
        
        # Configuration
        'vision_model': "qwen/qwen2.5-vl-72b-instruct:free",
        'llm_model': "meta-llama/llama-3.2-3b-instruct:free",
        'openrouter_key': '',
        'usda_key': '',
        'confidence_threshold': 0.7,
        'enable_caching': True,
        'debug_mode': False,
        'show_advanced': False,
        
        # UI State
        'show_ingredient_details': True,
        'chart_type': 'pie',
        'export_format': 'pdf'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_analysis():
    """Reset analysis-related session state for new analysis."""
    analysis_keys = [
        'analysis_stage',
        'thread_id',
        'uploaded_image_path',
        'uploaded_image_name',
        'original_analysis',
        'final_analysis',
        'enhanced_analysis',
        'validation_result',
        'validation_prompt',
        'natural_response',
        'wants_usda_info',
        'usda_matches',
        'modifications'
    ]
    
    # Reset analysis state
    st.session_state.analysis_stage = 'upload'
    st.session_state.thread_id = str(uuid.uuid4())
    
    # Clear analysis data
    for key in analysis_keys[2:]:  # Skip stage and thread_id
        if key in st.session_state:
            st.session_state[key] = None if 'usda_matches' not in key and 'wants_usda' not in key else (0 if 'matches' in key else False)
    
    # Clear modifications
    st.session_state.modifications = {}

def get_analysis_progress():
    """Get current analysis progress as percentage."""
    stage = st.session_state.get('analysis_stage', 'upload')
    
    progress_map = {
        'upload': 0,
        'analyzing': 25,
        'validating': 50,
        'modifying': 60,
        'usda_offer': 70,
        'enhancing': 85,
        'complete': 100
    }
    
    return progress_map.get(stage, 0)

def is_analysis_complete():
    """Check if analysis is complete."""
    return st.session_state.get('analysis_stage') == 'complete'

def has_valid_analysis():
    """Check if we have a valid analysis to work with."""
    return st.session_state.get('original_analysis') is not None

def get_current_analysis():
    """Get the most current analysis (enhanced > final > original)."""
    return (st.session_state.get('enhanced_analysis') or 
            st.session_state.get('final_analysis') or 
            st.session_state.get('original_analysis'))

def save_modification(ingredient_name: str, field: str, value):
    """Save a modification to session state."""
    if 'modifications' not in st.session_state:
        st.session_state.modifications = {}
    
    if ingredient_name not in st.session_state.modifications:
        st.session_state.modifications[ingredient_name] = {}
    
    st.session_state.modifications[ingredient_name][field] = value

def get_modification(ingredient_name: str, field: str, default=None):
    """Get a modification from session state."""
    return st.session_state.get('modifications', {}).get(ingredient_name, {}).get(field, default)

def clear_modifications():
    """Clear all modifications."""
    st.session_state.modifications = {}