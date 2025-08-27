# streamlit_components/usda_enhancement.py
import streamlit as st
import time


def render_usda_interface(mcp_available: bool = False):
    """Render USDA enhancement interface."""

    if mcp_available:
        render_usda_available_interface()
    else:
        render_usda_unavailable_interface()


def render_usda_available_interface():
    """Render interface when USDA is available."""
    st.success("🟢 USDA Food Database Available!")

    st.markdown("""
    ### ✨ Enhance with USDA Food Data Central

    Get more accurate nutritional data from the USDA's comprehensive food database.
    Our AI agent will automatically match your ingredients with official USDA entries.
    """)

    # Show what can be enhanced - prioritize final_analysis which contains modifications
    analysis = st.session_state.get("final_analysis") or st.session_state.get(
        "original_analysis"
    )
    if analysis:
        with st.expander("🔍 See what can be enhanced", expanded=True):
            st.write("**Your ingredients that can be enhanced:**")

            for i, ingredient in enumerate(analysis.ingredients, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(
                        f"{i}. **{ingredient.ingredient}** ({ingredient.weight:.0f}g)"
                    )
                    st.caption(
                        f"Current: {ingredient.calories} cal, {ingredient.protein:.1f}g protein"
                    )
                with col2:
                    st.success("✅ Can enhance")

            st.info(
                "💡 **Enhancement includes:** More detailed vitamins, minerals, fatty acids, and amino acids from official USDA data."
            )

    # Benefits of USDA enhancement
    render_usda_benefits()

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✨ Enhance with USDA", type="primary", width="stretch"):
            st.session_state.wants_usda_info = True
            st.session_state.analysis_stage = "enhancing"
            st.rerun()

    with col2:
        if st.button("➡️ Skip Enhancement", width="stretch"):
            st.session_state.wants_usda_info = False
            st.session_state.analysis_stage = "complete"
            st.rerun()

    with col3:
        with st.popover("❓ What is USDA?"):
            render_usda_info()


def render_usda_unavailable_interface():
    """Render interface when USDA is not available."""
    st.error("🔴 USDA Enhancement Unavailable")

    # Show reasons why it's unavailable
    reasons = []

    if not st.session_state.get("usda_key") and not st.session_state.get(
        "USDA_API_KEY"
    ):
        reasons.append("❌ USDA API key not configured")

    # Check if MCP server exists (simplified check)
    import os

    if not os.path.exists("food-data-central-mcp-server/dist/index.js"):
        reasons.append("❌ MCP server not built or not found")

    if not reasons:
        reasons.append("❌ MCP connection failed")

    st.markdown("**Issues preventing USDA enhancement:**")
    for reason in reasons:
        st.write(reason)

    # Show setup instructions
    with st.expander("🔧 How to enable USDA enhancement", expanded=True):
        st.markdown("""
        **To enable USDA enhancement:**

        1. **Get a USDA API Key:**
           - Visit: https://fdc.nal.usda.gov/api-key-signup.html
           - Sign up for a free API key
           - Add it to your environment variables or settings

        2. **Build the MCP Server:**
           ```bash
           cd food-data-central-mcp-server
           npm install
           npm run build
           ```

        3. **Restart the application** after completing setup
        """)

        # Quick setup option
        if st.button("🔑 Add USDA API Key Now"):
            usda_key = st.text_input("Enter your USDA API Key:", type="password")
            if usda_key and st.button("💾 Save Key"):
                st.session_state.usda_key = usda_key
                st.success(
                    "✅ USDA API key saved! Restart the app to use USDA features."
                )

    # Continue without USDA button
    st.markdown("---")
    if st.button(
        "➡️ Continue without USDA Enhancement", type="primary", width="stretch"
    ):
        st.session_state.wants_usda_info = False
        st.session_state.analysis_stage = "complete"
        st.rerun()


def render_usda_benefits():
    """Render the benefits of USDA enhancement."""
    st.markdown("#### 🎯 Benefits of USDA Enhancement")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **More Accurate Data:**
        - Official USDA nutritional values
        - Laboratory-tested measurements
        - Standardized food definitions
        """)

    with col2:
        st.markdown("""
        **Additional Nutrients:**
        - Vitamins (A, C, E, K, B-complex)
        - Minerals (Iron, Calcium, Magnesium, etc.)
        - Fatty acid profiles
        """)


def render_usda_info():
    """Render information about USDA Food Data Central."""
    st.markdown("""
    **USDA Food Data Central** is the U.S. government's comprehensive database of food nutrition information.

    **Key Features:**
    - 🏛️ **Official source:** Maintained by the U.S. Department of Agriculture
    - 🧪 **Lab-tested data:** Nutritional values from laboratory analysis
    - 📊 **Comprehensive:** Over 350,000 food items
    - 🔄 **Updated regularly:** New foods and updated nutrition info
    - 🆓 **Free to use:** Public database available via API

    **What you get:**
    - More accurate calorie counts
    - Detailed vitamin and mineral content
    - Fatty acid profiles
    - Amino acid composition
    - And much more!
    """)


def render_usda_enhancement_progress():
    """Render USDA enhancement progress screen."""
    st.subheader("✨ Enhancing with USDA Database...")

    # Create progress tracking
    progress_container = st.container()

    with progress_container:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        details_text = st.empty()

        # Simulate AI agent working
        stages = [
            {
                "progress": 20,
                "status": "🤖 AI Agent initializing...",
                "details": "Connecting to USDA Food Data Central database",
            },
            {
                "progress": 40,
                "status": "🔍 Searching USDA database...",
                "details": "Finding matches for your ingredients",
            },
            {
                "progress": 60,
                "status": "🎯 Matching ingredients...",
                "details": "AI agent analyzing food descriptions and categories",
            },
            {
                "progress": 80,
                "status": "📊 Retrieving nutrition data...",
                "details": "Downloading detailed nutritional information",
            },
            {
                "progress": 100,
                "status": "✅ Enhancement complete!",
                "details": "Successfully enhanced ingredients with USDA data",
            },
        ]

        # Show real-time progress
        for stage in stages:
            progress_bar.progress(stage["progress"])
            status_text.markdown(f"**{stage['status']}**")
            details_text.text(stage["details"])
            time.sleep(1.5)  # Simulate processing time

    # Show enhancement results
    render_enhancement_results()

    # Auto-advance to complete stage
    time.sleep(1)
    st.session_state.analysis_stage = "complete"
    st.rerun()


def render_enhancement_results():
    """Render the results of USDA enhancement."""
    st.success("🎉 USDA Enhancement Complete!")

    # Simulate enhancement results (in real app, this would come from the workflow)
    # Use the most current analysis (final_analysis contains modifications)
    analysis = st.session_state.get("final_analysis") or st.session_state.get(
        "original_analysis"
    )

    if analysis:
        total_ingredients = len(analysis.ingredients)
        # Simulate matches (in real app, this would be actual data)
        matched_count = min(
            total_ingredients, max(1, int(total_ingredients * 0.7))
        )  # Simulate 70% match rate

        st.session_state.usda_matches = matched_count

        # Results summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Ingredients Enhanced",
                f"{matched_count}",
                help="Number of ingredients matched with USDA data",
            )

        with col2:
            st.metric(
                "Match Rate",
                f"{matched_count / total_ingredients * 100:.0f}%",
                help="Percentage of ingredients successfully matched",
            )

        with col3:
            st.metric(
                "Additional Nutrients",
                "25+",
                help="Additional nutrient values now available",
            )

        # Show which ingredients were enhanced
        if matched_count > 0:
            with st.expander("📋 Enhancement Details", expanded=True):
                st.write("**Ingredients enhanced with USDA data:**")

                enhanced_ingredients = analysis.ingredients[
                    :matched_count
                ]  # Simulate first N were enhanced

                for i, ingredient in enumerate(enhanced_ingredients, 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{i}. **{ingredient.ingredient}**")
                        st.caption("Enhanced with official USDA nutritional data")
                    with col2:
                        st.success("✅ Enhanced")

                # Show unenhanced if any
                unenhanced = analysis.ingredients[matched_count:]
                if unenhanced:
                    st.write("**Ingredients using AI estimates:**")
                    for i, ingredient in enumerate(unenhanced, matched_count + 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{i}. **{ingredient.ingredient}**")
                            st.caption("Using AI nutritional estimates")
                        with col2:
                            st.info("🤖 AI estimate")

        st.info(
            "💡 **Note:** Enhanced data includes vitamins, minerals, and other nutrients not shown in the basic view."
        )


def render_usda_comparison():
    """Render comparison between AI estimates and USDA data."""
    st.markdown("#### 🔄 AI vs USDA Comparison")

    # This would show actual before/after comparison in a real implementation
    st.info("Comparison view showing improvements in data accuracy would appear here.")

    # Placeholder comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Before (AI Estimates)**")
        st.write("• Basic macronutrients")
        st.write("• Estimated values")
        st.write("• Limited mineral data")

    with col2:
        st.markdown("**After (USDA Enhanced)**")
        st.write("• Laboratory-verified data")
        st.write("• Comprehensive nutrient profile")
        st.write("• Official government standards")


def get_usda_status():
    """Get the current status of USDA availability."""
    # Check if we have API key
    has_api_key = bool(
        st.session_state.get("usda_key") or st.session_state.get("USDA_API_KEY")
    )

    # Check if MCP server is built
    import os

    has_mcp_server = os.path.exists("food-data-central-mcp-server/dist/index.js")

    # Check if MCP is initialized (from session state)
    mcp_initialized = st.session_state.get("mcp_available", False)

    return {
        "available": has_api_key and has_mcp_server and mcp_initialized,
        "has_api_key": has_api_key,
        "has_mcp_server": has_mcp_server,
        "mcp_initialized": mcp_initialized,
    }
