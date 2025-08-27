# streamlit_components/results_dashboard.py
import streamlit as st
import pandas as pd
from typing import Optional
import json


def render_results_dashboard(analysis):
    """Render the complete results dashboard."""
    if not analysis:
        st.error("No analysis data to display")
        return

    print("\n=== RESULTS DASHBOARD DATA ===")
    print(f"Analysis type: {type(analysis)}")
    print(f"Dish name: {analysis.dish_name}")
    print(f"Total calories: {analysis.total_calories}")
    print(f"Ingredients count: {len(analysis.ingredients)}")
    print(f"Total protein: {analysis.total_protein}")
    print(f"Total carbs: {analysis.total_carbohydrates}")
    print(f"Total fat: {analysis.total_fat}")
    print(f"Fiber: {getattr(analysis, 'total_fiber', 'N/A')}")
    print(f"Sodium: {getattr(analysis, 'total_sodium', 'N/A')}")
    print(f"Sugar: {getattr(analysis, 'total_sugar', 'N/A')}")
    for i, ing in enumerate(analysis.ingredients):
        print(
            f"  Ingredient {i + 1}: {ing.ingredient} - {ing.calories}cal, {ing.protein}g protein, {ing.carbohydrates}g carbs, {ing.fat}g fat"
        )
    print("=== END DASHBOARD DATA ===\n")

    # Hero section
    render_hero_section(analysis)

    # Main dashboard content
    render_main_dashboard(analysis)

    # Export and sharing options
    render_export_options(analysis)


def render_hero_section(analysis):
    """Render the hero section with main metrics."""
    st.markdown("# 🎉 Analysis Complete!")

    # Enhancement badge if applicable
    if st.session_state.get("usda_matches", 0) > 0:
        matches = st.session_state.get("usda_matches", 0)
        total_ingredients = len(analysis.ingredients)
        st.success(
            f"✨ Enhanced with USDA data: {matches}/{total_ingredients} ingredients"
        )

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🔥 Total Calories",
            value=f"{analysis.total_calories:,}",
            help="Total calories in this serving",
        )

    with col2:
        st.metric(
            label="🥩 Protein",
            value=f"{analysis.total_protein:.1f}g",
            help=f"Protein content ({(analysis.total_protein * 4 / analysis.total_calories * 100):.0f}% of calories)",
        )

    with col3:
        st.metric(
            label="🌾 Carbohydrates",
            value=f"{analysis.total_carbohydrates:.1f}g",
            help=f"Carb content ({(analysis.total_carbohydrates * 4 / analysis.total_calories * 100):.0f}% of calories)",
        )

    with col4:
        st.metric(
            label="🧈 Fat",
            value=f"{analysis.total_fat:.1f}g",
            help=f"Fat content ({(analysis.total_fat * 9 / analysis.total_calories * 100):.0f}% of calories)",
        )

    # Additional metrics if available
    if any([analysis.total_fiber, analysis.total_sugar, analysis.total_sodium]):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if analysis.total_fiber:
                st.metric(
                    label="🌱 Fiber",
                    value=f"{analysis.total_fiber:.1f}g",
                    help="Dietary fiber content",
                )

        with col2:
            if analysis.total_sugar:
                st.metric(
                    label="🍯 Sugar",
                    value=f"{analysis.total_sugar:.1f}g",
                    help="Total sugar content",
                )

        with col3:
            if analysis.total_sodium:
                st.metric(
                    label="🧂 Sodium",
                    value=f"{analysis.total_sodium:.0f}mg",
                    help="Sodium content",
                )

        with col4:
            total_weight = sum(ing.weight for ing in analysis.ingredients)
            st.metric(
                label="⚖️ Total Weight",
                value=f"{total_weight:.0f}g",
                help="Combined weight of all ingredients",
            )


def render_main_dashboard(analysis):
    """Render the main dashboard with tabs."""

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "🥕 Ingredients",
            "🔬 Micronutrients",
            "📈 Charts",
            "🔍 Analysis",
        ]
    )

    with tab1:
        render_overview_tab(analysis)

    with tab2:
        render_ingredients_tab(analysis)

    with tab3:
        render_micronutrients_section(analysis)

    with tab4:
        render_charts_tab(analysis)

    with tab5:
        render_analysis_tab(analysis)


def render_micronutrients_section(analysis):
    """Render micronutrients from USDA data if available."""
    st.markdown("### 🔬 Detailed Micronutrients (USDA Enhanced)")

    # Check if any ingredients have complete nutrition data
    enhanced_ingredients = [
        ing for ing in analysis.ingredients if ing.complete_nutrition
    ]

    if not enhanced_ingredients:
        st.info("💡 Micronutrient data available after USDA enhancement")
        return

    # Display micronutrients from all enhanced ingredients
    for ingredient in enhanced_ingredients:
        if ingredient.complete_nutrition:
            st.markdown(f"#### {ingredient.ingredient}")

            # Display nutrients in a more organized way
            nutrients = ingredient.complete_nutrition.nutrients

            # Group nutrients by category
            vitamins = {}
            minerals = {}
            other_nutrients = {}

            for nutrient_id, nutrient in nutrients.items():
                name = nutrient.name.lower()

                if any(
                    vitamin in name
                    for vitamin in [
                        "vitamin",
                        "folate",
                        "niacin",
                        "riboflavin",
                        "thiamin",
                        "choline",
                        "betaine",
                    ]
                ):
                    vitamins[nutrient.name] = nutrient
                elif any(
                    mineral in name
                    for mineral in [
                        "calcium",
                        "iron",
                        "magnesium",
                        "phosphorus",
                        "potassium",
                        "sodium",
                        "zinc",
                        "copper",
                        "manganese",
                        "selenium",
                    ]
                ):
                    minerals[nutrient.name] = nutrient
                elif nutrient.name.lower() not in [
                    "energy",
                    "protein",
                    "total lipid (fat)",
                    "carbohydrate, by difference",
                ]:
                    other_nutrients[nutrient.name] = nutrient

            # Display vitamins
            if vitamins:
                st.markdown("**🍊 Vitamins**")
                cols = st.columns(3)
                for i, (name, nutrient) in enumerate(vitamins.items()):
                    with cols[i % 3]:
                        st.metric(
                            label=name,
                            value=f"{nutrient.amount:.2f} {nutrient.unit}",
                            help=f"Per {ingredient.weight}g serving",
                        )

            # Display minerals
            if minerals:
                st.markdown("**⚡ Minerals**")
                cols = st.columns(3)
                for i, (name, nutrient) in enumerate(minerals.items()):
                    with cols[i % 3]:
                        st.metric(
                            label=name,
                            value=f"{nutrient.amount:.2f} {nutrient.unit}",
                            help=f"Per {ingredient.weight}g serving",
                        )

            # Display other nutrients
            if other_nutrients:
                st.markdown("**📊 Other Nutrients**")
                cols = st.columns(3)
                for i, (name, nutrient) in enumerate(other_nutrients.items()):
                    with cols[i % 3]:
                        st.metric(
                            label=name,
                            value=f"{nutrient.amount:.2f} {nutrient.unit}",
                            help=f"Per {ingredient.weight}g serving",
                        )

            st.divider()


def render_overview_tab(analysis):
    """Render the overview tab."""
    col1, col2 = st.columns([2, 1])

    with col1:
        # Dish information
        st.markdown("### 🍽️ Dish Information")

        info_data = {
            "Dish Name": analysis.dish_name,
            "Portion Size": analysis.portion_size,
            "Total Ingredients": len(analysis.ingredients),
            "Analysis Confidence": f"{analysis.confidence_score:.1%}"
            if analysis.confidence_score
            else "N/A",
        }

        for label, value in info_data.items():
            st.write(f"**{label}:** {value}")

        # Nutritional highlights
        st.markdown("### 🌟 Nutritional Highlights")

        highlights = generate_nutritional_highlights(analysis)
        for highlight in highlights:
            if highlight.startswith("✅"):
                st.success(highlight)
            elif highlight.startswith("⚠️"):
                st.warning(highlight)
            else:
                st.info(highlight)

    with col2:
        # Quick stats
        st.markdown("### 📊 Quick Stats")

        total_cal = analysis.total_calories
        if total_cal > 0:
            protein_pct = (analysis.total_protein * 4) / total_cal * 100
            carb_pct = (analysis.total_carbohydrates * 4) / total_cal * 100
            fat_pct = (analysis.total_fat * 9) / total_cal * 100

            st.write("**Macronutrient Distribution:**")
            st.write(f"• Protein: {protein_pct:.0f}%")
            st.write(f"• Carbs: {carb_pct:.0f}%")
            st.write(f"• Fat: {fat_pct:.0f}%")

            # Calorie density
            total_weight = sum(ing.weight for ing in analysis.ingredients)
            if total_weight > 0:
                cal_density = total_cal / total_weight * 100  # cal per 100g
                st.write(f"**Calorie Density:** {cal_density:.0f} cal/100g")

        # Health category
        render_health_category(analysis)


def render_ingredients_tab(analysis):
    """Render the ingredients breakdown tab."""
    st.markdown("### 🥕 Ingredient Breakdown")

    # Create detailed ingredients table
    data = []
    for i, ing in enumerate(analysis.ingredients, 1):
        # Calculate percentage of total calories
        cal_percent = (
            (ing.calories / analysis.total_calories * 100)
            if analysis.total_calories > 0
            else 0
        )

        data.append(
            {
                "Rank": i,
                "Ingredient": ing.ingredient,
                "Weight (g)": f"{ing.weight:.0f}",
                "Calories": f"{ing.calories:,}",
                "% of Total": f"{cal_percent:.1f}%",
                "Protein (g)": f"{ing.protein:.1f}",
                "Carbs (g)": f"{ing.carbohydrates:.1f}",
                "Fat (g)": f"{ing.fat:.1f}",
                "Cal/100g": f"{(ing.calories / ing.weight * 100):.0f}"
                if ing.weight > 0
                else "0",
            }
        )

    df = pd.DataFrame(data)

    # Display interactive table
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Ingredient": st.column_config.TextColumn("Ingredient", width="large"),
            "Weight (g)": st.column_config.TextColumn("Weight (g)", width="small"),
            "Calories": st.column_config.TextColumn("Calories", width="small"),
            "% of Total": st.column_config.TextColumn("% of Total", width="small"),
            "Cal/100g": st.column_config.TextColumn(
                "Cal/100g", width="small", help="Calorie density"
            ),
        },
    )

    # Top contributors
    st.markdown("#### 🏆 Top Calorie Contributors")

    sorted_ingredients = sorted(
        analysis.ingredients, key=lambda x: x.calories, reverse=True
    )

    col1, col2, col3 = st.columns(3)
    for i, ing in enumerate(sorted_ingredients[:3]):
        col = [col1, col2, col3][i]
        with col:
            cal_percent = (
                (ing.calories / analysis.total_calories * 100)
                if analysis.total_calories > 0
                else 0
            )
            st.metric(
                label=f"#{i + 1} {ing.ingredient}",
                value=f"{ing.calories} cal",
                delta=f"{cal_percent:.1f}% of total",
            )


def render_charts_tab(analysis):
    """Render the charts and visualizations tab."""
    st.markdown("### 📈 Visual Analysis")

    # Chart type selector
    chart_type = st.selectbox(
        "Select chart type:",
        [
            "Macronutrient Pie Chart",
            "Calorie Distribution",
            "Ingredient Comparison",
            "Nutrient Profile",
        ],
        index=0,
    )

    if chart_type == "Macronutrient Pie Chart":
        render_macro_pie_chart(analysis)
    elif chart_type == "Calorie Distribution":
        render_calorie_distribution_chart(analysis)
    elif chart_type == "Ingredient Comparison":
        render_ingredient_comparison_chart(analysis)
    elif chart_type == "Nutrient Profile":
        render_nutrient_profile_chart(analysis)


def render_macro_pie_chart(analysis):
    """Render macronutrient pie chart."""
    try:
        import plotly.express as px

        # Calculate macro calories
        protein_cal = analysis.total_protein * 4
        carb_cal = analysis.total_carbohydrates * 4
        fat_cal = analysis.total_fat * 9

        # Create pie chart data
        values = [protein_cal, carb_cal, fat_cal]
        labels = ["Protein", "Carbohydrates", "Fat"]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        fig = px.pie(
            values=values,
            names=labels,
            title="Macronutrient Distribution (by calories)",
            color_discrete_sequence=colors,
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Calories: %{value}<br>Percentage: %{percent}<extra></extra>",
        )

        fig.update_layout(showlegend=True, height=500)

        st.plotly_chart(fig, width="stretch")

    except ImportError:
        st.warning("📊 Install plotly to see interactive charts: `pip install plotly`")
        render_simple_macro_breakdown(analysis)


def render_calorie_distribution_chart(analysis):
    """Render calorie distribution by ingredient."""
    try:
        import plotly.express as px

        # Prepare data
        ingredients = [ing.ingredient for ing in analysis.ingredients]
        calories = [ing.calories for ing in analysis.ingredients]

        fig = px.bar(
            x=calories,
            y=ingredients,
            orientation="h",
            title="Calorie Distribution by Ingredient",
            labels={"x": "Calories", "y": "Ingredient"},
            color=calories,
            color_continuous_scale="Viridis",
        )

        fig.update_layout(height=max(400, len(ingredients) * 40))

        st.plotly_chart(fig, width="stretch")

    except ImportError:
        # Fallback to simple bar chart
        st.write("**Calorie Distribution:**")
        for ing in sorted(analysis.ingredients, key=lambda x: x.calories, reverse=True):
            percent = (
                (ing.calories / analysis.total_calories * 100)
                if analysis.total_calories > 0
                else 0
            )
            st.progress(
                percent / 100,
                text=f"{ing.ingredient}: {ing.calories} cal ({percent:.1f}%)",
            )


def render_ingredient_comparison_chart(analysis):
    """Render ingredient comparison chart."""
    st.write("**Ingredient Nutrition Comparison**")

    # Create comparison data
    comparison_data = []
    for ing in analysis.ingredients:
        comparison_data.append(
            {
                "Ingredient": ing.ingredient,
                "Calories": ing.calories,
                "Protein": ing.protein,
                "Carbs": ing.carbohydrates,
                "Fat": ing.fat,
                "Weight": ing.weight,
            }
        )

    df = pd.DataFrame(comparison_data)

    # Allow user to select what to compare
    metric = st.selectbox(
        "Compare ingredients by:", ["Calories", "Protein", "Carbs", "Fat", "Weight"]
    )

    if len(df) > 0:
        fig_data = df.sort_values(metric, ascending=False)
        st.bar_chart(fig_data.set_index("Ingredient")[metric])


def render_nutrient_profile_chart(analysis):
    """Render nutrient profile radar chart."""
    st.write("**Nutritional Profile Overview**")

    # Simple nutrient overview
    nutrients = {
        "Calories": analysis.total_calories,
        "Protein (g)": analysis.total_protein,
        "Carbs (g)": analysis.total_carbohydrates,
        "Fat (g)": analysis.total_fat,
        "Fiber (g)": analysis.total_fiber or 0,
        "Sodium (mg)": (analysis.total_sodium or 0)
        / 10,  # Scale down for visualization
    }

    col1, col2 = st.columns(2)

    with col1:
        for i, (nutrient, value) in enumerate(list(nutrients.items())[:3]):
            st.metric(nutrient, f"{value:.1f}")

    with col2:
        for i, (nutrient, value) in enumerate(list(nutrients.items())[3:]):
            if nutrient == "Sodium (mg)":
                st.metric("Sodium (mg)", f"{value * 10:.0f}")  # Scale back up
            else:
                st.metric(nutrient, f"{value:.1f}")


def render_analysis_tab(analysis):
    """Render detailed analysis tab."""
    st.markdown("### 🔍 Detailed Analysis")

    # Analysis metadata
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Analysis Details")
        st.write(
            f"**Analysis Method:** {'USDA Enhanced' if st.session_state.get('usda_matches', 0) > 0 else 'AI Vision Analysis'}"
        )
        st.write(
            f"**Confidence Score:** {analysis.confidence_score:.1%}"
            if analysis.confidence_score
            else "N/A"
        )
        st.write(f"**Number of Ingredients:** {len(analysis.ingredients)}")

        if st.session_state.get("usda_matches", 0) > 0:
            matches = st.session_state.get("usda_matches", 0)
            st.write(
                f"**USDA Matches:** {matches}/{len(analysis.ingredients)} ingredients"
            )

    with col2:
        st.markdown("#### 🎯 Quality Assessment")

        quality_score = calculate_quality_score(analysis)
        st.metric("Overall Quality", f"{quality_score}/10")

        # Quality factors
        factors = assess_quality_factors(analysis)
        for factor, score in factors.items():
            emoji = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
            st.write(f"{emoji} **{factor}:** {score}/10")

    # Raw data export
    with st.expander("📋 Raw Analysis Data", expanded=False):
        st.json(analysis.model_dump())


def render_export_options(analysis):
    """Render export and sharing options."""
    st.markdown("---")
    st.markdown("### 📤 Export & Share")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📄 Download PDF Report", width="stretch"):
            pdf_data = generate_pdf_report(analysis)
            if pdf_data:
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_data,
                    file_name=f"nutrition_report_{analysis.dish_name.lower().replace(' ', '_')}.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("PDF generation not available")

    with col2:
        csv_data = generate_csv_report(analysis)
        st.download_button(
            label="📊 Download CSV Data",
            data=csv_data,
            file_name=f"nutrition_data_{analysis.dish_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch",
        )

    with col3:
        json_data = json.dumps(analysis.model_dump(), indent=2)
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name=f"nutrition_analysis_{analysis.dish_name.lower().replace(' ', '_')}.json",
            mime="application/json",
            width="stretch",
        )

    with col4:
        if st.button("📱 Share Results", width="stretch"):
            render_share_options(analysis)


def generate_nutritional_highlights(analysis) -> list:
    """Generate nutritional highlights for the analysis."""
    highlights = []

    total_cal = analysis.total_calories

    # Calorie assessment
    if total_cal < 300:
        highlights.append("✅ Low calorie option - great for weight management")
    elif total_cal > 800:
        highlights.append("⚠️ High calorie content - consider portion control")

    # Protein assessment
    if total_cal > 0:
        protein_percent = (analysis.total_protein * 4) / total_cal * 100
        if protein_percent > 25:
            highlights.append("✅ High protein content - excellent for muscle building")
        elif protein_percent < 10:
            highlights.append("⚠️ Low protein - consider adding protein sources")

    # Fiber assessment
    if analysis.total_fiber and analysis.total_fiber > 5:
        highlights.append("✅ Good fiber content - supports digestive health")

    # Sodium assessment
    if analysis.total_sodium and analysis.total_sodium > 1500:
        highlights.append("⚠️ High sodium - monitor daily salt intake")

    return highlights


def render_health_category(analysis):
    """Render health category assessment."""
    st.markdown("**Health Category:**")

    score = calculate_quality_score(analysis)

    if score >= 8:
        st.success("🌟 Excellent nutritional profile")
    elif score >= 6:
        st.info("👍 Good nutritional balance")
    elif score >= 4:
        st.warning("⚖️ Moderate nutritional value")
    else:
        st.error("⚠️ Consider nutritional improvements")


def calculate_quality_score(analysis) -> int:
    """Calculate overall nutritional quality score (1-10)."""
    score = 5  # Base score

    total_cal = analysis.total_calories

    if total_cal > 0:
        # Protein factor
        protein_percent = (analysis.total_protein * 4) / total_cal * 100
        if protein_percent >= 15:
            score += 1
        elif protein_percent < 10:
            score -= 1

        # Fiber factor
        if analysis.total_fiber and analysis.total_fiber >= 5:
            score += 1

        # Sodium factor
        if analysis.total_sodium:
            if analysis.total_sodium <= 600:
                score += 1
            elif analysis.total_sodium > 1500:
                score -= 1

        # Calorie factor
        if 300 <= total_cal <= 600:
            score += 1
        elif total_cal > 800:
            score -= 1

    return max(1, min(10, score))


def assess_quality_factors(analysis) -> dict:
    """Assess individual quality factors."""
    factors = {}

    # Protein quality
    total_cal = analysis.total_calories
    if total_cal > 0:
        protein_percent = (analysis.total_protein * 4) / total_cal * 100
        if protein_percent >= 20:
            factors["Protein Content"] = 9
        elif protein_percent >= 15:
            factors["Protein Content"] = 7
        elif protein_percent >= 10:
            factors["Protein Content"] = 5
        else:
            factors["Protein Content"] = 3

    # Fiber quality
    if analysis.total_fiber:
        if analysis.total_fiber >= 8:
            factors["Fiber Content"] = 9
        elif analysis.total_fiber >= 5:
            factors["Fiber Content"] = 7
        elif analysis.total_fiber >= 3:
            factors["Fiber Content"] = 5
        else:
            factors["Fiber Content"] = 3
    else:
        factors["Fiber Content"] = 2

    # Sodium quality
    if analysis.total_sodium:
        if analysis.total_sodium <= 400:
            factors["Sodium Level"] = 9
        elif analysis.total_sodium <= 800:
            factors["Sodium Level"] = 7
        elif analysis.total_sodium <= 1200:
            factors["Sodium Level"] = 5
        else:
            factors["Sodium Level"] = 2
    else:
        factors["Sodium Level"] = 8  # Assume good if no sodium data

    return factors


def generate_pdf_report(analysis) -> Optional[bytes]:
    """Generate PDF report (placeholder - requires reportlab or similar)."""
    # This would require additional dependencies like reportlab
    # For now, return None to indicate PDF generation is not available
    return None


def generate_csv_report(analysis) -> str:
    """Generate CSV report of the analysis."""
    data = []

    # Summary row
    data.append(
        {
            "Type": "Summary",
            "Item": analysis.dish_name,
            "Weight_g": sum(ing.weight for ing in analysis.ingredients),
            "Calories": analysis.total_calories,
            "Protein_g": analysis.total_protein,
            "Carbs_g": analysis.total_carbohydrates,
            "Fat_g": analysis.total_fat,
            "Fiber_g": analysis.total_fiber or 0,
            "Sugar_g": analysis.total_sugar or 0,
            "Sodium_mg": analysis.total_sodium or 0,
        }
    )

    # Individual ingredients
    for ing in analysis.ingredients:
        data.append(
            {
                "Type": "Ingredient",
                "Item": ing.ingredient,
                "Weight_g": ing.weight,
                "Calories": ing.calories,
                "Protein_g": ing.protein,
                "Carbs_g": ing.carbohydrates,
                "Fat_g": ing.fat,
                "Fiber_g": ing.fiber or 0,
                "Sugar_g": ing.sugar or 0,
                "Sodium_mg": ing.sodium or 0,
            }
        )

    # Convert to CSV
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def render_share_options(analysis):
    """Render sharing options."""
    st.info("🔗 Sharing functionality coming soon!")

    # Placeholder for sharing features
    st.write("**Share via:**")
    st.write("• Generate shareable link (coming soon)")
    st.write("• Social media integration (coming soon)")
    st.write("• Email report (coming soon)")


def render_simple_macro_breakdown(analysis):
    """Simple fallback for macro breakdown without plotly."""
    st.write("**Macronutrient Breakdown (by calories):**")

    protein_cal = analysis.total_protein * 4
    carb_cal = analysis.total_carbohydrates * 4
    fat_cal = analysis.total_fat * 9
    total_macro_cal = protein_cal + carb_cal + fat_cal

    if total_macro_cal > 0:
        st.write(
            f"🥩 **Protein:** {protein_cal:.0f} cal ({protein_cal / total_macro_cal * 100:.0f}%)"
        )
        st.progress(protein_cal / total_macro_cal)

        st.write(
            f"🌾 **Carbohydrates:** {carb_cal:.0f} cal ({carb_cal / total_macro_cal * 100:.0f}%)"
        )
        st.progress(carb_cal / total_macro_cal)

        st.write(
            f"🧈 **Fat:** {fat_cal:.0f} cal ({fat_cal / total_macro_cal * 100:.0f}%)"
        )
        st.progress(fat_cal / total_macro_cal)
