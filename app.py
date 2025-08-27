# app.py - Main Streamlit application for Calorie Analyzer
import streamlit as st
import os
import uuid
import time
import logging
from dotenv import load_dotenv

# Import our existing workflow
from calorie_app.workflows.nutrition_workflow import NutritionWorkflow
from calorie_app.models.nutrition import UserModification
from calorie_app.utils.logging_config import standardized_logger

# Import Streamlit components (we'll create these)
from streamlit_components.session_manager import init_session_state, reset_analysis
from streamlit_components.image_uploader import render_image_uploader
from streamlit_components.analysis_display import render_analysis_display
from streamlit_components.modification_ui import render_modification_interface
from streamlit_components.usda_enhancement import render_usda_interface
from streamlit_components.results_dashboard import render_results_dashboard
from streamlit_components.sidebar_config import render_sidebar_config

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="🍽️ AI Nutrition Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize standardized logging
standardized_logger.setup_logging()
logger = logging.getLogger(__name__)

def setup_workflow():
    """Initialize the nutrition workflow with current settings."""
    try:
        # Get API keys from session state or environment
        config = {
            "vision_model": st.session_state.get("vision_model", "qwen/qwen2.5-vl-72b-instruct:free"),
            "vision_api_key": st.session_state.get("openrouter_key") or os.getenv("OPENROUTER_API_KEY"),
            "vision_base_url": os.getenv("OPENROUTER_API_URL"),
            "llm_model": st.session_state.get("llm_model", "meta-llama/llama-3.2-3b-instruct:free"),
            "llm_api_key": st.session_state.get("openrouter_key") or os.getenv("OPENROUTER_API_KEY"),
            "llm_base_url": os.getenv("OPENROUTER_API_URL")
        }
        
        # Setup MCP config for USDA if available
        mcp_config = None
        usda_api_key = st.session_state.get("usda_key") or os.getenv("USDA_API_KEY")
        if usda_api_key:
            mcp_server_path = "food-data-central-mcp-server/dist/index.js"
            if os.path.exists(mcp_server_path):
                mcp_config = {
                    "food-data-central": {
                        "command": "node",
                        "args": [mcp_server_path],
                        "env": {"USDA_API_KEY": usda_api_key},
                        "transport": "stdio"
                    }
                }
        
        # Create workflow instance
        workflow = NutritionWorkflow(
            vision_model=config["vision_model"],
            vision_api_key=config["vision_api_key"],
            vision_base_url=config["vision_base_url"],
            llm_model=config["llm_model"],
            llm_api_key=config["llm_api_key"],
            llm_base_url=config["llm_base_url"],
            mcp_config=mcp_config
        )
        
        st.session_state.workflow_obj = workflow
        st.session_state.mcp_available = mcp_config is not None
        return workflow
        
    except Exception as e:
        st.error(f"❌ Failed to initialize workflow: {str(e)}")
        logger.error(f"Workflow initialization error: {str(e)}")
        return None

def render_header():
    """Render the main header with status indicators."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🍽️ AI Nutrition Analyzer")
        st.caption("Analyze food images with AI • Human validation • USDA enhancement")
    
    with col2:
        # Status indicators
        stage = st.session_state.get('analysis_stage', 'upload')
        stage_info = {
            'upload': ('📤', 'Ready to analyze'),
            'analyzing': ('⏳', 'Analyzing image...'),
            'validating': ('✋', 'Awaiting validation'),
            'enhancing': ('✨', 'USDA enhancement'),
            'complete': ('✅', 'Analysis complete')
        }
        
        icon, status = stage_info.get(stage, ('❓', 'Unknown'))
        st.metric("Status", f"{icon} {status}")
    
    with col3:
        # Quick actions
        if st.button("🔄 Start New Analysis", type="secondary"):
            reset_analysis()
            st.rerun()
        
        if st.button("💾 Save Analysis", disabled=stage != 'complete'):
            st.success("Analysis saved! (Feature coming soon)")

def main():
    """Main application logic."""
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar configuration
    render_sidebar_config()
    
    # Render main header
    render_header()
    
    # Setup workflow if not already done
    if 'workflow_obj' not in st.session_state or st.session_state.workflow_obj is None:
        workflow = setup_workflow()
        if not workflow:
            st.stop()
    else:
        workflow = st.session_state.workflow_obj
    
    # Main content area based on analysis stage
    stage = st.session_state.get('analysis_stage', 'upload')
    
    if stage == 'upload':
        render_upload_stage()
        
    elif stage == 'analyzing':
        render_analyzing_stage()
        
    elif stage == 'validating':
        render_validating_stage()
        
    elif stage == 'modifying':
        render_modification_stage()
        
    elif stage == 'usda_offer':
        render_usda_offer_stage()
        
    elif stage == 'enhancing':
        render_enhancing_stage()
        
    elif stage == 'complete':
        render_complete_stage()
    
    # Debug information (if enabled)
    if st.session_state.get('debug_mode', False):
        render_debug_info()

def render_upload_stage():
    """Render the image upload interface."""
    st.subheader("📤 Upload Food Image")
    
    # Use the image uploader component
    uploaded_file = render_image_uploader()
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_uploads/{uuid.uuid4()}_{uploaded_file.name}"
        os.makedirs("temp_uploads", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_image_path = temp_path
        st.session_state.uploaded_image_name = uploaded_file.name
        
        # Show analyze button
        if st.button("🚀 Analyze Image", type="primary", width="stretch"):
            st.session_state.analysis_stage = 'analyzing'
            st.rerun()

def render_analyzing_stage():
    """Render the analysis progress screen."""
    st.subheader("⏳ Analyzing Your Food Image...")
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run analysis
    try:
        image_path = st.session_state.uploaded_image_path
        thread_id = st.session_state.thread_id
        workflow = st.session_state.workflow_obj
        
        
        # Update progress
        status_text.text("🖼️ Processing image...")
        progress_bar.progress(25)
        
        # Start analysis (this calls the workflow)
        status_text.text("🤖 Analyzing with AI vision model...")
        progress_bar.progress(50)
        
        # Run the analysis
        result = workflow.start_analysis(image_path, thread_id)
        
        status_text.text("🧮 Calculating nutrition values...")
        progress_bar.progress(75)
        
        if result.get('success'):
            st.session_state.original_analysis = result.get('original_analysis')
            st.session_state.validation_result = result.get('validation_result')
            st.session_state.validation_prompt = result.get('validation_prompt')
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            time.sleep(1)  # Brief pause to show completion
            
            st.session_state.analysis_stage = 'validating'
            st.rerun()
            
        else:
            st.error(f"❌ Analysis failed: {result.get('error_message', 'Unknown error')}")
            st.session_state.analysis_stage = 'upload'
            
    except Exception as e:
        st.error(f"💥 Analysis error: {str(e)}")
        logger.error(f"Analysis stage error: {str(e)}")
        st.session_state.analysis_stage = 'upload'
        
    # Cancel button
    if st.button("❌ Cancel Analysis"):
        st.session_state.analysis_stage = 'upload'
        st.rerun()

def render_validating_stage():
    """Render the validation and approval interface."""
    st.subheader("✋ Review Analysis Results")
    
    analysis = st.session_state.get('original_analysis')
    if not analysis:
        st.error("No analysis data found")
        st.session_state.analysis_stage = 'upload'
        st.rerun()
        return
    
    # Display analysis results
    render_analysis_display(analysis, st.session_state.get('validation_result'))
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Approve Analysis", type="primary", width="stretch"):
            # Submit validation without modifications
            submit_validation(approved=True, modifications=None)
    
    with col2:
        if st.button("✏️ Modify Analysis", type="secondary", width="stretch"):
            st.session_state.analysis_stage = 'modifying'
            st.rerun()
    
    with col3:
        if st.button("❌ Reject & Restart", width="stretch"):
            reset_analysis()
            st.rerun()

def render_modification_stage():
    """Render the modification interface."""
    st.subheader("✏️ Modify Analysis")
    
    analysis = st.session_state.get('original_analysis')
    modifications = render_modification_interface(analysis)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Apply Changes", type="primary", width="stretch"):
            submit_validation(approved=True, modifications=modifications)
    
    with col2:
        if st.button("↩️ Back to Review", width="stretch"):
            st.session_state.analysis_stage = 'validating'
            st.rerun()

def render_usda_offer_stage():
    """Render USDA enhancement offer."""
    st.subheader("✨ Enhance with USDA Data")
    
    mcp_available = st.session_state.get('mcp_available', False)
    render_usda_interface(mcp_available)

def render_enhancing_stage():
    """Render USDA enhancement progress."""
    st.subheader("✨ Enhancing with USDA Database...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Get the workflow and run USDA enhancement
        workflow = st.session_state.workflow_obj
        thread_id = st.session_state.thread_id
        
        # Simulate progress stages
        stages = [
            (25, "🤖 AI Agent searching USDA database..."),
            (50, "🔍 Matching ingredients with food entries..."),
            (75, "📊 Retrieving detailed nutrition data..."),
        ]
        
        for progress, message in stages:
            status_text.text(message)
            progress_bar.progress(progress)
            time.sleep(1)
        
        # Actually run the USDA enhancement
        status_text.text("🔄 Processing USDA enhancement...")
        progress_bar.progress(90)
        
        # Re-submit validation with USDA enhancement enabled
        result = workflow.submit_human_validation(
            thread_id=thread_id,
            approved=True,
            modifications=None,
            wants_usda_info=True
        )
        
        if result.get('success'):
            # Update with enhanced data
            enhanced_analysis = result.get('dish_analysis')
            usda_matches = result.get('usda_matches_count', 0)
            natural_response = result.get('natural_response')
            
            st.session_state.enhanced_analysis = enhanced_analysis
            st.session_state.usda_matches = usda_matches
            st.session_state.natural_response = natural_response
            
            print("\n=== USDA ENHANCEMENT COMPLETE ===")
            print(f"Enhanced analysis: {enhanced_analysis is not None}")
            print(f"USDA matches: {usda_matches}")
            print(f"Natural response: {natural_response is not None}")
            if enhanced_analysis:
                print(f"Enhanced ingredients count: {len(enhanced_analysis.ingredients)}")
                for i, ing in enumerate(enhanced_analysis.ingredients):
                    print(f"  {i+1}. {ing.ingredient}: {ing.calories}cal, {ing.protein}g protein, {ing.carbohydrates}g carbs, {ing.fat}g fat")
            print("=== END ENHANCEMENT DATA ===\n")
            
            status_text.text("✅ Enhancement complete!")
            progress_bar.progress(100)
            time.sleep(1)
            
            st.session_state.analysis_stage = 'complete'
            st.rerun()
        else:
            st.error(f"❌ USDA enhancement failed: {result.get('error_message')}")
            # Fall back to complete stage without enhancement
            st.session_state.analysis_stage = 'complete'
            st.rerun()
            
    except Exception as e:
        st.error(f"💥 Enhancement error: {str(e)}")
        logger.error(f"USDA enhancement error: {str(e)}")
        # Fall back to complete stage
        st.session_state.analysis_stage = 'complete'
        st.rerun()
    
    # Cancel button
    if st.button("❌ Cancel Enhancement"):
        st.session_state.analysis_stage = 'complete'
        st.rerun()

def render_complete_stage():
    """Render the final results dashboard."""
    st.subheader("🎉 Analysis Complete!")
    
    # Display the natural language response first
    natural_response = st.session_state.get('natural_response')
    if natural_response:
        st.subheader("📝 Analysis Summary")
        st.write(natural_response)
        st.divider()
    
    # Get final analysis (enhanced or original)
    enhanced_analysis = st.session_state.get('enhanced_analysis')
    final_analysis = st.session_state.get('final_analysis')
    original_analysis = st.session_state.get('original_analysis')
    usda_matches = st.session_state.get('usda_matches', 0)
    
    print("\n=== RENDER COMPLETE STAGE ===")
    print(f"Enhanced analysis: {enhanced_analysis is not None}")
    print(f"Final analysis: {final_analysis is not None}")
    print(f"Original analysis: {original_analysis is not None}")
    print(f"USDA matches: {usda_matches}")
    print("=== END RENDER INFO ===\n")
    
    display_analysis = (enhanced_analysis or final_analysis or original_analysis)
    
    if display_analysis:
        render_results_dashboard(display_analysis)
    else:
        st.error("No analysis data available")

def submit_validation(approved: bool, modifications: UserModification = None):
    """Submit human validation and continue workflow."""
    try:
        workflow = st.session_state.workflow_obj
        thread_id = st.session_state.thread_id
        
        # First, submit validation without USDA info to get the base analysis
        result = workflow.submit_human_validation(
            thread_id=thread_id,
            approved=approved,
            modifications=modifications,
            wants_usda_info=False  # Always start with False
        )
        
        if result.get('success'):
            st.session_state.final_analysis = result.get('dish_analysis')
            st.session_state.natural_response = result.get('natural_response')
            
            # Print the natural response to console for debugging
            natural_response = result.get('natural_response')
            if natural_response:
                print(f"\n=== FINAL NATURAL LANGUAGE RESPONSE ===\n{natural_response}\n=== END RESPONSE ===")
            
            # Always go to USDA offer stage if MCP is available
            if st.session_state.get('mcp_available'):
                st.session_state.analysis_stage = 'usda_offer'
            else:
                st.session_state.analysis_stage = 'complete'
                
            st.rerun()
        else:
            st.error(f"❌ Validation failed: {result.get('error_message')}")
            
    except Exception as e:
        st.error(f"💥 Validation error: {str(e)}")
        logger.error(f"Validation submission error: {str(e)}")

def render_debug_info():
    """Render debug information."""
    with st.expander("🐛 Debug Information", expanded=False):
        st.write("**Session State:**")
        debug_state = {k: str(v) for k, v in st.session_state.items() 
                      if not k.startswith('workflow_obj')}  # Don't show workflow object
        st.json(debug_state)
        
        st.write("**System Info:**")
        st.write(f"- Thread ID: {st.session_state.get('thread_id', 'Not set')}")
        st.write(f"- MCP Available: {st.session_state.get('mcp_available', False)}")
        st.write(f"- Analysis Stage: {st.session_state.get('analysis_stage', 'upload')}")

if __name__ == "__main__":
    main()