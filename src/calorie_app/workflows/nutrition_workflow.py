# workflows/nutrition_workflow.py
from typing import Dict, Any, Optional
import asyncio
import logging
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from calorie_app.models.nutrition import WorkflowState, AnalysisState, UserModification
from calorie_app.services.vision_service import VisionService
from calorie_app.services.nutrition_service import NutritionService
from calorie_app.services.formatting_service import FormattingService
from calorie_app.utils.config import ModelDefaults
from calorie_app.utils.logging_config import log_workflow_step

logger = logging.getLogger(__name__)

class NutritionWorkflow:
    """Complete nutrition analysis workflow with human-in-the-loop validation."""
    
    def __init__(
        self,
        vision_model: str = ModelDefaults.VISION_MODEL,
        vision_api_key: str = None,
        vision_base_url: str = None,
        llm_model: str = ModelDefaults.LLM_MODEL,
        llm_api_key: str = None,
        llm_base_url: str = None,
        mcp_config: Optional[Dict] = None,
        # AI Agent specific configuration
        agent_model: Optional[str] = None,
        agent_api_key: Optional[str] = None,
        agent_base_url: Optional[str] = None
    ):
        """Initialize workflow with all required services."""
        
        # Initialize services
        self.vision_service = VisionService(
            model_name=vision_model,
            api_key=vision_api_key,
            base_url=vision_base_url
        )
        
        self.nutrition_service = NutritionService(
            mcp_config=mcp_config,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            # Pass agent-specific parameters
            agent_model=agent_model or llm_model,  # Default to main LLM if not specified
            agent_api_key=agent_api_key or llm_api_key,
            agent_base_url=agent_base_url or llm_base_url
        )
        
        self.formatting_service = FormattingService(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url
        )
        
        # Initialize MCP service if configured
        self._mcp_initialized = False
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        self.memory = MemorySaver()
        
        # Compile workflow with checkpointer
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with human validation and USDA query."""
        
        workflow = StateGraph(WorkflowState)
        
        # Add workflow nodes
        workflow.add_node("initialize_mcp", self._initialize_mcp_node)
        workflow.add_node("analyze_image", self._analyze_image_node)
        workflow.add_node("validate_analysis", self._validate_analysis_node)
        workflow.add_node("await_human_validation", self._await_human_validation_node)
        workflow.add_node("apply_modifications", self._apply_modifications_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("offer_usda_query", self._offer_usda_query_node)
        workflow.add_node("handle_usda_query", self._handle_usda_query_node)
        workflow.add_node("finalize_response", self._finalize_response_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define workflow edges
        workflow.set_entry_point("initialize_mcp")
        
        workflow.add_edge("initialize_mcp", "analyze_image")
        workflow.add_edge("analyze_image", "validate_analysis")
        
        # After validation, always wait for human input
        workflow.add_conditional_edges(
            "validate_analysis",
            self._should_continue_after_validation,
            {
                "await_human": "await_human_validation",
                "error": "handle_error"
            }
        )
        
        # Human validation decision point
        workflow.add_conditional_edges(
            "await_human_validation",
            self._route_human_decision,
            {
                "apply_modifications": "apply_modifications",
                "generate_response": "generate_response",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("apply_modifications", "generate_response")
        
        # After generating response, offer USDA query
        workflow.add_edge("generate_response", "offer_usda_query")
        
        # USDA query decision point
        workflow.add_conditional_edges(
            "offer_usda_query",
            self._should_handle_usda_query,
            {
                "handle_usda": "handle_usda_query",
                "finalize": "finalize_response"
            }
        )
        
        workflow.add_edge("handle_usda_query", "finalize_response")
        workflow.add_edge("finalize_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow
    
    def _initialize_mcp_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Initialize MCP connections."""
        try:
            if not self._mcp_initialized:
                # Run async initialization in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(self.nutrition_service.initialize_mcp())
                    self._mcp_initialized = success
                    logger.info(f"MCP initialization: {'successful' if success else 'failed'}")
                finally:
                    loop.close()
            
            return {"mcp_initialized": self._mcp_initialized}
            
        except Exception as e:
            logger.warning(f"MCP initialization failed: {str(e)}")
            return {"mcp_initialized": False}
    
    def _analyze_image_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Analyze food image using vision model."""
        try:
            log_workflow_step("Image Analysis", "Starting", f"image: {state.image_path}", logger)
            
            if not state.image_path:
                return {
                    "status": AnalysisState.FAILED,
                    "error_message": "No image path provided"
                }
            
            # Analyze image with VLM
            analysis_result = self.vision_service.analyze_dish_image(state.image_path)
            
            if not analysis_result:
                state.retry_count += 1
                if state.retry_count >= 2:
                    return {
                        "status": AnalysisState.FAILED,
                        "error_message": "Vision analysis failed after 2 attempts"
                    }
                else:
                    logger.warning(f"Retrying image analysis (attempt {state.retry_count})")
                    return {"retry_count": state.retry_count}
            
            log_workflow_step("Image Analysis", "Complete", f"identified: {analysis_result.dish_name}", logger)
            return {
                "original_analysis": analysis_result,
                "status": AnalysisState.VALIDATION,
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"[ERR] Image analysis failed: {str(e)}")
            return {
                "status": AnalysisState.FAILED,
                "error_message": f"Analysis failed: {str(e)}"
            }
    
    def _validate_analysis_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Validate analysis and generate validation report."""
        try:
            log_workflow_step("Validation", "Starting", "", logger)
            
            if not state.original_analysis:
                return {
                    "status": AnalysisState.FAILED,
                    "error_message": "No analysis to validate"
                }
            
            # Validate the analysis
            validation_result = self.nutrition_service.validate_nutrition_data(
                state.original_analysis
            )
            
            log_workflow_step("Validation", "Complete", f"confidence: {validation_result.confidence_score:.2f}", logger)
            
            return {
                "validation_result": validation_result,
                "status": AnalysisState.COMPLETE if validation_result.is_valid else AnalysisState.FAILED
            }
            
        except Exception as e:
            logger.error(f"[ERR] Validation failed: {str(e)}")
            return {
                "status": AnalysisState.FAILED,
                "error_message": f"Validation failed: {str(e)}"
            }
    
    def _apply_modifications_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Apply user modifications to the analysis."""
        try:
            log_workflow_step("Modifications", "Applying", "", logger)
            
            if not state.user_modifications:
                # No modifications - use original analysis
                return {
                    "final_analysis": state.original_analysis,
                    "status": AnalysisState.COMPLETE
                }
            
            # Ensure we have original analysis to modify
            if not state.original_analysis:
                logger.error("No original analysis available for modification")
                return {
                    "status": AnalysisState.FAILED,
                    "error_message": "No original analysis available for modification"
                }
            
            # Convert user modifications to proper object if needed
            if isinstance(state.user_modifications, dict):
                user_mods = UserModification(**state.user_modifications)
            else:
                user_mods = state.user_modifications
            
            # Apply modifications (this only queries LLM for new ingredients)
            modified_analysis = self.nutrition_service.apply_user_modifications(
                state.original_analysis,
                user_mods
            )
            
            log_workflow_step("Modifications", "Complete", "", logger)
            return {
                "final_analysis": modified_analysis,
                "status": AnalysisState.COMPLETE
            }
            
        except Exception as e:
            logger.error(f"[ERR] Modifications failed: {str(e)}")
            return {
                "status": AnalysisState.FAILED,
                "error_message": f"Failed to apply modifications: {str(e)}"
            }
    
    def _generate_response_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate final natural language response."""
        try:
            log_workflow_step("Response Generation", "Starting", "", logger)
            
            # Use final analysis if available, otherwise original
            analysis_to_use = state.final_analysis or state.original_analysis
            
            if not analysis_to_use:
                return {
                    "status": AnalysisState.FAILED,
                    "error_message": "No analysis available for response generation"
                }
            
            # Generate natural response (with confidence info)
            natural_response = self.formatting_service.generate_natural_response(
                analysis_to_use,
                state.validation_result
            )
            
            log_workflow_step("Response Generation", "Complete", "", logger)
            return {
                "natural_response": natural_response,
                "formatted_tone_response": natural_response,  # Same as natural_response for now
                "final_analysis": analysis_to_use,  # Set final_analysis to the analysis we used
                "status": AnalysisState.COMPLETE
            }
            
        except Exception as e:
            logger.error(f"[ERR] Response generation failed: {str(e)}")
            return {
                "status": AnalysisState.FAILED,
                "error_message": f"Response generation failed: {str(e)}"
            }
    
    def _handle_error_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Handle workflow errors."""
        logger.error(f"[ERR] Workflow error: {state.error_message}")
        return {
            "status": AnalysisState.FAILED,
            "natural_response": f"Analysis failed: {state.error_message or 'Unknown error'}"
        }
    
    def _await_human_validation_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Pause workflow for human validation input."""
        log_workflow_step("Human Validation", "Awaiting", "", logger)
        
        # Use new interrupt method
        from langgraph.types import interrupt
        
        # Create structured interrupt payload
        interrupt_payload = {
            "type": "human_validation_request",
            "dish_name": state.original_analysis.dish_name if state.original_analysis else "Unknown",
            "total_calories": state.original_analysis.total_calories if state.original_analysis else 0,
            "ingredients_count": len(state.original_analysis.ingredients) if state.original_analysis else 0,
            "ingredients": [{
                "name": ing.ingredient,
                "weight": ing.weight,
                "calories": ing.calories
            } for ing in state.original_analysis.ingredients] if state.original_analysis else [],
            "prompt": (
                f"Please review this nutrition analysis:\n\n"
                f"Dish: {state.original_analysis.dish_name if state.original_analysis else 'Unknown'}\n"
                f"Total Calories: {state.original_analysis.total_calories if state.original_analysis else 0}\n"
                f"Ingredients: {len(state.original_analysis.ingredients) if state.original_analysis else 0} items\n\n"
                f"You can:\n"
                f"1. Accept the analysis as-is\n"
                f"2. Modify the dish name, ingredient weights, or add/remove ingredients\n"
                f"3. Cancel the analysis\n\n"
                f"What would you like to do?"
            )
        }
        
        # Store interrupt data in state for later retrieval
        state.human_validation_prompt = interrupt_payload["prompt"]
        
        # Use new interrupt function to pause workflow
        interrupt(interrupt_payload["prompt"])
    
    def _offer_usda_query_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Offer user the option to query USDA database."""
        log_workflow_step("USDA Query", "Offering", "", logger)
        
        return {
            "usda_query_available": self._mcp_initialized,
            "usda_query_prompt": "Would you like to get more detailed nutritional information from the USDA database for any ingredients?"
        }
    
    def _handle_usda_query_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Handle user USDA database queries using AI agent for ingredient matching."""
        try:
            logger.info("AI Agent: Starting USDA ingredient matching")
            
            # Get the current analysis (final if modified, otherwise original)
            current_analysis = state.final_analysis or state.original_analysis
            if not current_analysis:
                logger.error("No analysis available for USDA enhancement")
                return {
                    "usda_query_handled": False,
                    "error_message": "No analysis available for USDA enhancement"
                }
            
            # Use AI agent to match ingredients with USDA database
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                logger.info(f"AI Agent: Matching {len(current_analysis.ingredients)} ingredients")
                
                enhanced_ingredients = loop.run_until_complete(
                    self.nutrition_service.match_ingredients_with_usda_efficient(
                        current_analysis.ingredients,
                        dish_context=current_analysis.dish_name
                    )
                )
                
                # Create enhanced analysis with USDA-matched ingredients
                enhanced_analysis = current_analysis.model_copy(deep=True)
                enhanced_analysis.ingredients = enhanced_ingredients
                
                # Recalculate totals with enhanced data
                enhanced_analysis.calculate_totals()
                
                # Count successful matches
                usda_matches = sum(1 for ing in enhanced_ingredients if hasattr(ing, 'complete_nutrition') and ing.complete_nutrition)
                
                logger.info(f"AI Agent: USDA matching complete - {usda_matches}/{len(enhanced_ingredients)} ingredients enhanced")
                
                return {
                    "final_analysis": enhanced_analysis,
                    "usda_enhanced_analysis": enhanced_analysis,
                    "usda_matches_count": usda_matches,
                    "total_ingredients": len(enhanced_ingredients),
                    "usda_query_handled": True,
                    "usda_enhancement_summary": f"Enhanced {usda_matches} of {len(enhanced_ingredients)} ingredients with USDA data using AI agent"
                }
                
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"💥 AI Agent USDA query error: {str(e)}")
            return {
                "usda_query_handled": False,
                "error_message": f"AI Agent USDA query failed: {str(e)}"
            }
    
    def _finalize_response_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Finalize the complete workflow response."""
        logger.info("Finalizing workflow response")
        
        # Use USDA-enhanced analysis if available, otherwise use final/original
        final_analysis = (state.usda_enhanced_analysis or 
                         state.final_analysis or 
                         state.original_analysis)
        
        finalization_data = {
            "status": AnalysisState.COMPLETE,
            "workflow_complete": True,
            "final_analysis": final_analysis
        }
        
        # Add USDA enhancement summary if available
        if hasattr(state, 'usda_matches_count') and state.usda_matches_count is not None:
            finalization_data.update({
                "usda_enhanced": True,
                "usda_matches_count": state.usda_matches_count,
                "total_ingredients": state.total_ingredients,
                "usda_enhancement_summary": state.usda_enhancement_summary
            })
            logger.info(f"Workflow finalized with USDA enhancement: {state.usda_matches_count}/{state.total_ingredients} ingredients enhanced")
        else:
            finalization_data["usda_enhanced"] = False
            logger.info("Workflow finalized with VLM/LLM data only")
        
        return finalization_data
    
    def _generate_validation_prompt(self, analysis) -> str:
        """Generate a prompt for human validation."""
        if not analysis:
            return "Please review the analysis and provide any modifications needed."
        
        prompt = f"""
Please review this nutrition analysis:

Dish: {analysis.dish_name}
Total Calories: {analysis.total_calories}
Ingredients: {len(analysis.ingredients)} items

You can:
1. Accept the analysis as-is
2. Modify the dish name, ingredient weights, or add/remove ingredients
3. Cancel the analysis

What would you like to do?
"""
        return prompt.strip()
    
    # Conditional routing functions
    def _should_continue_after_validation(self, state: WorkflowState) -> str:
        """Route after initial validation."""
        logger.debug(f"[WORKFLOW] Validation routing: status={state.status}")
        if state.status == AnalysisState.FAILED:
            logger.debug("[WORKFLOW] -> Routing to error handler")
            return "error"
        logger.debug("[WORKFLOW] -> Routing to await human validation")
        return "await_human"
    
    def _route_human_decision(self, state: WorkflowState) -> str:
        """Route based on human validation decision."""
        logger.debug(f"[WORKFLOW] Human decision routing: status={state.status}")
        if state.status == AnalysisState.FAILED:
            logger.debug("[WORKFLOW] -> Routing to error handler")
            return "error"
        
        # Check if user provided modifications (handle both dict and object forms)
        has_modifications = False
        if state.user_modifications:
            # Handle both UserModification object and dict forms
            if isinstance(state.user_modifications, dict):
                # Check if any modification fields have values
                mods = state.user_modifications
                has_modifications = (
                    mods.get('dish_name') or 
                    mods.get('ingredients_to_add') or 
                    mods.get('ingredients_to_remove') or 
                    mods.get('ingredient_weight_changes') or
                    mods.get('portion_size')
                )
                logger.debug(f"[WORKFLOW] User modifications (dict): {mods}")
            else:
                # UserModification object
                has_modifications = True
                logger.debug(f"[WORKFLOW] User modifications (object): {type(state.user_modifications)}")
        
        if has_modifications:
            logger.info("[WORKFLOW] -> User has modifications, routing to apply_modifications")
            return "apply_modifications"
        else:
            logger.info("[WORKFLOW] -> No modifications, routing to generate_response")
            return "generate_response"
    
    def _should_handle_usda_query(self, state: WorkflowState) -> str:
        """Decide whether to handle USDA queries."""
        wants_usda = hasattr(state, 'wants_usda_info') and state.wants_usda_info
        logger.debug(f"[WORKFLOW] USDA query routing: wants_usda={wants_usda}, mcp_initialized={self._mcp_initialized}")
        
        # Check if user wants USDA information and MCP is available
        if wants_usda and self._mcp_initialized:
            logger.debug("[WORKFLOW] -> Routing to handle USDA query")
            return "handle_usda"
        logger.debug("[WORKFLOW] -> Routing to finalize")
        return "finalize"
    
    def analyze_dish(
        self, 
        image_path: str, 
        modifications: Optional[UserModification] = None,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Run complete dish analysis workflow.
        
        Args:
            image_path: Path to food image
            modifications: Optional user modifications
            thread_id: Thread ID for workflow persistence
            
        Returns:
            Complete analysis results
        """
        try:
            log_workflow_step("Workflow", "Starting", f"image: {Path(image_path).name}", logger)
            
            # Initial state
            initial_state = WorkflowState(
                image_path=image_path,
                user_modifications=modifications,
                status=AnalysisState.PENDING
            )
            
            # Run workflow
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.app.invoke(initial_state.model_dump(), config=config)
            
            # Convert back to WorkflowState
            result_state = WorkflowState(**final_state)
            
            # Prepare results
            results = {
                "success": result_state.status == AnalysisState.COMPLETE,
                "status": result_state.status,
                "dish_analysis": result_state.final_analysis or result_state.original_analysis,
                "natural_response": result_state.natural_response,
                "validation_result": result_state.validation_result,
                "error_message": result_state.error_message
            }
            
            # Add health insights if successful
            if results["success"] and results["dish_analysis"]:
                results["health_insights"] = self.formatting_service.generate_health_insights(
                    results["dish_analysis"]
                )
            
            log_workflow_step("Workflow", "Completed", f"status: {result_state.status}", logger)
            return results
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return {
                "success": False,
                "status": AnalysisState.FAILED,
                "error_message": f"Workflow failed: {str(e)}",
                "dish_analysis": None,
                "natural_response": "Analysis failed due to system error.",
                "validation_result": None,
                "health_insights": []
            }
    
    def get_workflow_state(self, thread_id: str = "default") -> Optional[WorkflowState]:
        """Get current state of a workflow thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state_snapshot = self.app.get_state(config)
            
            if state_snapshot and state_snapshot.values:
                return WorkflowState(**state_snapshot.values)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting workflow state: {str(e)}")
            return None
    
    def resume_workflow_with_modifications(
        self,
        modifications: UserModification,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Resume paused workflow with user modifications.
        
        Args:
            modifications: User modifications to apply
            thread_id: Thread ID of paused workflow
            
        Returns:
            Updated analysis results
        """
        try:
            logger.info("Resuming workflow with user modifications")
            
            # Update state with modifications
            config = {"configurable": {"thread_id": thread_id}}
            current_state = self.app.get_state(config)
            
            if not current_state:
                raise ValueError("No workflow state found for thread")
            
            # Update with modifications
            updated_state = current_state.values.copy()
            updated_state["user_modifications"] = modifications.model_dump()
            
            # Continue workflow
            final_state = self.app.invoke(updated_state, config=config)
            result_state = WorkflowState(**final_state)
            
            # Generate modification summary
            if (result_state.final_analysis and 
                result_state.original_analysis and 
                result_state.user_modifications):
                
                modification_summary = self.formatting_service.generate_modification_summary(
                    result_state.original_analysis,
                    result_state.final_analysis
                )
            else:
                modification_summary = None
            
            return {
                "success": result_state.status == AnalysisState.COMPLETE,
                "status": result_state.status,
                "dish_analysis": result_state.final_analysis,
                "natural_response": result_state.natural_response,
                "modification_summary": modification_summary,
                "health_insights": self.formatting_service.generate_health_insights(
                    result_state.final_analysis
                ) if result_state.final_analysis else []
            }
            
        except Exception as e:
            logger.error(f"Resume workflow error: {str(e)}")
            return {
                "success": False,
                "error_message": f"Failed to resume workflow: {str(e)}"
            }
    
    def start_analysis(self, image_path: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Start analysis workflow and pause at human validation.
        
        Args:
            image_path: Path to food image
            thread_id: Thread ID for workflow persistence
            
        Returns:
            Initial analysis results with validation prompt
        """
        try:
            log_workflow_step("Analysis", "Starting", f"image: {Path(image_path).name}", logger)
            
            # Initial state
            initial_state = WorkflowState(
                image_path=image_path,
                status=AnalysisState.PENDING
            )
            
            # Run workflow until interrupt
            config = {"configurable": {"thread_id": thread_id}}
            
            # Execute workflow with streaming to handle interrupts
            final_state = None
            interrupted = False
            
            for step in self.app.stream(initial_state.model_dump(), config=config):
                if '__interrupt__' in step:
                    # Workflow was interrupted - get current state
                    current_state = self.app.get_state(config)
                    if current_state and current_state.values:
                        return {
                            "success": True,
                            "awaiting_validation": True,
                            "validation_prompt": current_state.values.get("human_validation_prompt", "Please review the analysis"),
                            "original_analysis": current_state.values.get("original_analysis"),
                            "validation_result": current_state.values.get("validation_result"),
                            "thread_id": thread_id
                        }
                    interrupted = True
                    break
                final_state = step
            
            if not interrupted and final_state:
                # Workflow completed without interruption
                return {
                    "success": True,
                    "awaiting_validation": False,
                    "validation_prompt": None,
                    "original_analysis": final_state.get("original_analysis"),
                    "validation_result": final_state.get("validation_result"),
                    "thread_id": thread_id
                }
            
        except Exception as e:
            logger.error(f"Analysis start error: {str(e)}")
            return {
                "success": False,
                "error_message": f"Failed to start analysis: {str(e)}"
            }
    
    def submit_human_validation(
        self,
        thread_id: str,
        approved: bool = True,
        modifications: Optional[UserModification] = None,
        wants_usda_info: bool = False
    ) -> Dict[str, Any]:
        """
        Submit human validation decision and continue workflow.
        
        Args:
            thread_id: Thread ID of paused workflow
            approved: Whether user approved the analysis
            modifications: Optional user modifications
            wants_usda_info: Whether user wants USDA database info
            
        Returns:
            Updated analysis results
        """
        try:
            log_workflow_step("Human Validation", "Submitting", f"approved: {approved}", logger)
            
            # Get current state to preserve existing data
            config = {"configurable": {"thread_id": thread_id}}
            current_state = self.app.get_state(config)
            
            if not current_state:
                raise ValueError("No workflow state found for thread")
            
            # Prepare the input for resuming, preserving existing state
            resume_input = current_state.values.copy()
            resume_input.update({
                "user_modifications": modifications,
                "wants_usda_info": wants_usda_info,
                "awaiting_human_input": False
            })
            
            if not approved:
                resume_input["status"] = AnalysisState.FAILED.value
                resume_input["error_message"] = "Analysis rejected by user"
            
            # Update the current state with resume input
            self.app.update_state(config, resume_input, as_node="await_human_validation")
            
            # Resume the interrupted workflow
            final_state = None
            
            # Stream the workflow to completion from where it left off
            for chunk in self.app.stream(None, config=config, stream_mode="values"):
                final_state = chunk
            
            if not final_state:
                raise ValueError("No final state received from workflow")
                
            result_state = WorkflowState(**final_state)
            
            return {
                "success": result_state.status == AnalysisState.COMPLETE,
                "status": result_state.status,
                "dish_analysis": (getattr(result_state, 'usda_enhanced_analysis', None) or 
                                result_state.final_analysis or 
                                result_state.original_analysis),
                "natural_response": result_state.natural_response,
                "formatted_tone_response": getattr(result_state, 'formatted_tone_response', ''),
                "usda_query_available": getattr(result_state, 'usda_query_available', False),
                "usda_query_prompt": getattr(result_state, 'usda_query_prompt', ''),
                "usda_tools": getattr(result_state, 'usda_tools_available', []),
                "usda_enhanced": getattr(result_state, 'usda_enhanced', False),
                "usda_matches_count": getattr(result_state, 'usda_matches_count', 0),
                "usda_enhancement_summary": getattr(result_state, 'usda_enhancement_summary', ''),
                "workflow_complete": getattr(result_state, 'workflow_complete', False)
            }
            
        except Exception as e:
            logger.error(f"Human validation submission error: {str(e)}")
            return {
                "success": False,
                "error_message": f"Failed to submit validation: {str(e)}"
            }
    
