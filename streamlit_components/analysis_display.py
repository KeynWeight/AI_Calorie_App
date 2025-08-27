# streamlit_components/analysis_display.py
import streamlit as st
import pandas as pd

def render_analysis_display(analysis, validation_result=None):
    """Render the analysis results display."""
    if not analysis:
        st.error("No analysis data to display")
        return
    
    # Hero metrics section
    render_hero_metrics(analysis)
    
    # Main content sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_ingredients_table(analysis.ingredients)
        
    with col2:
        render_nutrition_summary(analysis)
        if validation_result:
            render_confidence_info(validation_result)

def render_hero_metrics(analysis):
    """Render the main metrics at the top."""
    st.markdown("### 📊 Analysis Results")
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔥 Calories",
            value=f"{analysis.total_calories:,}",
            help="Total calories in this serving"
        )
    
    with col2:
        st.metric(
            label="🥩 Protein",
            value=f"{analysis.total_protein:.1f}g",
            help="Total protein content"
        )
    
    with col3:
        st.metric(
            label="🌾 Carbs",
            value=f"{analysis.total_carbohydrates:.1f}g",
            help="Total carbohydrates"
        )
    
    with col4:
        st.metric(
            label="🧈 Fat",
            value=f"{analysis.total_fat:.1f}g",
            help="Total fat content"
        )
    
    # Secondary metrics row
    if any([analysis.total_fiber, analysis.total_sugar, analysis.total_sodium]):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if analysis.total_fiber:
                st.metric(
                    label="🌱 Fiber",
                    value=f"{analysis.total_fiber:.1f}g",
                    help="Dietary fiber content"
                )
        
        with col2:
            if analysis.total_sugar:
                st.metric(
                    label="🍯 Sugar",
                    value=f"{analysis.total_sugar:.1f}g",
                    help="Total sugar content"
                )
        
        with col3:
            if analysis.total_sodium:
                st.metric(
                    label="🧂 Sodium",
                    value=f"{analysis.total_sodium:.0f}mg",
                    help="Sodium content"
                )
        
        with col4:
            # Calculate total weight
            total_weight = sum(ing.weight for ing in analysis.ingredients)
            st.metric(
                label="⚖️ Weight",
                value=f"{total_weight:.0f}g",
                help="Total weight of all ingredients"
            )

def render_ingredients_table(ingredients):
    """Render the ingredients breakdown table."""
    st.markdown("### 🥕 Ingredients Breakdown")
    
    if not ingredients:
        st.warning("No ingredients found")
        return
    
    # Create dataframe for the table
    data = []
    for i, ing in enumerate(ingredients, 1):
        data.append({
            '#': i,
            'Ingredient': ing.ingredient,
            'Weight (g)': f"{ing.weight:.0f}",
            'Calories': f"{ing.calories:,}",
            'Protein (g)': f"{ing.protein:.1f}",
            'Carbs (g)': f"{ing.carbohydrates:.1f}",
            'Fat (g)': f"{ing.fat:.1f}",
            'Fiber (g)': f"{ing.fiber:.1f}" if ing.fiber else "0.0",
            'Sugar (g)': f"{ing.sugar:.1f}" if ing.sugar else "0.0",
            'Sodium (mg)': f"{ing.sodium:.0f}" if ing.sodium else "0"
        })
    
    df = pd.DataFrame(data)
    
    # Display as interactive table
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            '#': st.column_config.NumberColumn('#', width="small"),
            'Ingredient': st.column_config.TextColumn('Ingredient', width="large"),
            'Weight (g)': st.column_config.TextColumn('Weight (g)', width="small"),
            'Calories': st.column_config.TextColumn('Calories', width="small"),
            'Protein (g)': st.column_config.TextColumn('Protein (g)', width="small"),
            'Carbs (g)': st.column_config.TextColumn('Carbs (g)', width="small"),
            'Fat (g)': st.column_config.TextColumn('Fat (g)', width="small")
        }
    )
    
    # Show detailed view toggle
    if st.checkbox("🔍 Show detailed breakdown", key="show_detailed_ingredients"):
        render_detailed_ingredients(ingredients)

def render_detailed_ingredients(ingredients):
    """Render detailed ingredient information."""
    st.markdown("#### Detailed Ingredient Analysis")
    
    for i, ing in enumerate(ingredients, 1):
        with st.expander(f"{i}. {ing.ingredient} ({ing.weight:.0f}g)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Macronutrients:**")
                st.write(f"• Calories: {ing.calories:,} kcal")
                st.write(f"• Protein: {ing.protein:.1f}g")
                st.write(f"• Carbohydrates: {ing.carbohydrates:.1f}g")
                st.write(f"• Fat: {ing.fat:.1f}g")
            
            with col2:
                st.write("**Other Nutrients:**")
                st.write(f"• Fiber: {ing.fiber:.1f}g" if ing.fiber else "• Fiber: 0.0g")
                st.write(f"• Sugar: {ing.sugar:.1f}g" if ing.sugar else "• Sugar: 0.0g")
                st.write(f"• Sodium: {ing.sodium:.0f}mg" if ing.sodium else "• Sodium: 0mg")
                
                # Calculate calories per gram
                cal_per_gram = ing.calories / ing.weight if ing.weight > 0 else 0
                st.write(f"• Density: {cal_per_gram:.1f} cal/g")

def render_nutrition_summary(analysis):
    """Render nutrition summary and insights."""
    st.markdown("### 📋 Nutrition Summary")
    
    # Basic info card
    with st.container():
        st.markdown(f"""
        **🍽️ Dish:** {analysis.dish_name}
        
        **📏 Portion:** {analysis.portion_size}
        
        **🎯 Confidence:** {analysis.confidence_score:.1%}
        """)
    
    # Macro breakdown chart
    if st.checkbox("📊 Show macro breakdown", value=True):
        render_macro_chart(analysis)
    
    # Health insights
    render_health_insights(analysis)

def render_macro_chart(analysis):
    """Render macronutrient breakdown chart."""
    try:
        import plotly.express as px
        
        # Calculate macro percentages
        total_cal = analysis.total_calories
        if total_cal > 0:
            protein_cal = analysis.total_protein * 4
            carb_cal = analysis.total_carbohydrates * 4
            fat_cal = analysis.total_fat * 9
            
            # Create pie chart
            labels = ['Protein', 'Carbohydrates', 'Fat']
            values = [protein_cal, carb_cal, fat_cal]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            fig = px.pie(
                values=values,
                names=labels,
                title="Macronutrient Breakdown (by calories)",
                color_discrete_sequence=colors
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=True, height=300)
            
            st.plotly_chart(fig, width="stretch")
            
            # Show percentages
            st.write("**Calorie Distribution:**")
            st.write(f"• Protein: {protein_cal:.0f} cal ({protein_cal/total_cal*100:.0f}%)")
            st.write(f"• Carbs: {carb_cal:.0f} cal ({carb_cal/total_cal*100:.0f}%)")
            st.write(f"• Fat: {fat_cal:.0f} cal ({fat_cal/total_cal*100:.0f}%)")
    
    except ImportError:
        # Fallback to simple text display
        st.write("**Macronutrient Breakdown:**")
        total_cal = analysis.total_calories
        if total_cal > 0:
            protein_cal = analysis.total_protein * 4
            carb_cal = analysis.total_carbohydrates * 4
            fat_cal = analysis.total_fat * 9
            
            st.write(f"• Protein: {protein_cal/total_cal*100:.0f}%")
            st.write(f"• Carbohydrates: {carb_cal/total_cal*100:.0f}%")
            st.write(f"• Fat: {fat_cal/total_cal*100:.0f}%")

def render_health_insights(analysis):
    """Render health insights based on the analysis."""
    st.markdown("#### 🌟 Nutritional Highlights")
    
    insights = []
    warnings = []
    tips = []
    
    # Calorie assessment - more detailed ranges
    if analysis.total_calories > 800:
        warnings.append("⚠️ High calorie content - consider portion control")
    elif analysis.total_calories > 600:
        insights.append("🔥 Moderate calorie content - good for main meals")
    elif analysis.total_calories < 200:
        insights.append("✅ Light meal option - perfect for snacks")
    else:
        insights.append("🎯 Balanced calorie content")
    
    # Protein assessment - more detailed
    total_cal = analysis.total_calories
    if total_cal > 0:
        protein_percent = (analysis.total_protein * 4) / total_cal * 100
        if protein_percent > 30:
            insights.append("💪 Very high protein - excellent for muscle building")
        elif protein_percent > 25:
            insights.append("💪 High protein content - great for muscle building")
        elif protein_percent > 15:
            insights.append("🥩 Good protein balance")
        elif protein_percent < 10:
            warnings.append("⚠️ Low protein - consider adding protein sources")
        else:
            tips.append("💡 Moderate protein content")
    
    # Carb assessment
    if total_cal > 0:
        carb_percent = (analysis.total_carbohydrates * 4) / total_cal * 100
        if carb_percent > 60:
            tips.append("🌾 High carb content - good for energy")
        elif carb_percent < 20:
            insights.append("🥬 Low carb option")
    
    # Fat assessment
    if total_cal > 0:
        fat_percent = (analysis.total_fat * 9) / total_cal * 100
        if fat_percent > 40:
            warnings.append("⚠️ High fat content - watch portion sizes")
        elif fat_percent > 30:
            tips.append("🧈 Moderate fat content")
        elif fat_percent < 15:
            insights.append("✅ Low fat option")
    
    # Fiber assessment - more detailed
    if analysis.total_fiber:
        if analysis.total_fiber > 8:
            insights.append("🌱 Excellent fiber content - great for digestive health")
        elif analysis.total_fiber > 5:
            insights.append("🌱 Good fiber content - supports digestive health")
        elif analysis.total_fiber > 2:
            tips.append("🌾 Some fiber content")
        else:
            tips.append("💡 Consider adding more fiber-rich foods")
    else:
        tips.append("💡 Add vegetables or whole grains for fiber")
    
    # Sugar assessment
    if analysis.total_sugar:
        if analysis.total_sugar > 25:
            warnings.append("🍯 High sugar content - enjoy in moderation")
        elif analysis.total_sugar > 15:
            tips.append("🍯 Moderate sugar content")
        elif analysis.total_sugar < 5:
            insights.append("✅ Low sugar content")
    
    # Sodium assessment - more detailed
    if analysis.total_sodium:
        if analysis.total_sodium > 1500:
            warnings.append("🧂 Very high sodium - watch your daily intake")
        elif analysis.total_sodium > 1000:
            warnings.append("🧂 High sodium content - monitor daily intake")
        elif analysis.total_sodium > 500:
            tips.append("🧂 Moderate sodium content")
        else:
            insights.append("✅ Low sodium option")
    else:
        insights.append("✅ No added sodium detected")
    
    # Weight-based insights
    total_weight = sum(ing.weight for ing in analysis.ingredients) if analysis.ingredients else 0
    if total_weight > 0:
        calorie_density = analysis.total_calories / total_weight
        if calorie_density > 4:
            tips.append("⚖️ High calorie density - smaller portions recommended")
        elif calorie_density < 1.5:
            insights.append("⚖️ Low calorie density - filling and satisfying")
    
    # Display all insights
    for insight in insights[:3]:  # Show max 3 insights
        st.success(insight)
    
    for warning in warnings[:2]:  # Show max 2 warnings
        st.warning(warning)
    
    for tip in tips[:2]:  # Show max 2 tips
        st.info(tip)
    
    # Always show at least something
    if not insights and not warnings and not tips:
        st.info("💡 This appears to be a well-balanced meal!")

def render_confidence_info(validation_result):
    """Render confidence and validation information."""
    if not validation_result:
        return
    
    st.markdown("### 🎯 Analysis Confidence")
    
    # Confidence meter
    confidence = validation_result.confidence_score
    
    # Set status based on confidence
    if confidence >= 0.8:
        status = "High"
    elif confidence >= 0.6:
        status = "Medium"
    else:
        status = "Low"
    
    st.metric(
        label="Confidence Level",
        value=f"{confidence:.1%}",
        help=f"{status} confidence in the analysis accuracy"
    )
    
    # Progress bar
    st.progress(confidence)
    
    # Show warnings if any
    if hasattr(validation_result, 'warnings') and validation_result.warnings:
        st.warning("**Warnings:**")
        for warning in validation_result.warnings:
            st.write(f"• {warning}")
    
    # Show suggestions if any
    if hasattr(validation_result, 'suggestions') and validation_result.suggestions:
        st.info("**Suggestions:**")
        for suggestion in validation_result.suggestions:
            st.write(f"• {suggestion}")

def render_comparison_view(original_analysis, modified_analysis):
    """Render before/after comparison view."""
    if not original_analysis or not modified_analysis:
        return
    
    st.markdown("### 🔄 Before vs After Comparison")
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Before (Original)")
        st.metric("Calories", f"{original_analysis.total_calories:,}")
        st.metric("Protein", f"{original_analysis.total_protein:.1f}g")
        st.metric("Carbs", f"{original_analysis.total_carbohydrates:.1f}g")
        st.metric("Fat", f"{original_analysis.total_fat:.1f}g")
    
    with col2:
        st.markdown("#### After (Modified)")
        cal_delta = modified_analysis.total_calories - original_analysis.total_calories
        protein_delta = modified_analysis.total_protein - original_analysis.total_protein
        carb_delta = modified_analysis.total_carbohydrates - original_analysis.total_carbohydrates
        fat_delta = modified_analysis.total_fat - original_analysis.total_fat
        
        st.metric("Calories", f"{modified_analysis.total_calories:,}", delta=f"{cal_delta:+,}")
        st.metric("Protein", f"{modified_analysis.total_protein:.1f}g", delta=f"{protein_delta:+.1f}g")
        st.metric("Carbs", f"{modified_analysis.total_carbohydrates:.1f}g", delta=f"{carb_delta:+.1f}g")
        st.metric("Fat", f"{modified_analysis.total_fat:.1f}g", delta=f"{fat_delta:+.1f}g")