# streamlit_components/modification_ui.py
import streamlit as st
from calorie_app.models.nutrition import UserModification
from typing import Optional, Dict, List


def render_modification_interface(analysis) -> Optional[UserModification]:
    """Render the modification interface and return UserModification object."""
    if not analysis:
        st.error("No analysis data to modify")
        return None

    st.markdown("### ✏️ Modify Your Analysis")
    st.write(
        "Make changes to dish name, ingredients, or weights. Changes will be processed by our AI system."
    )

    modifications = {}

    # Dish name modification
    with st.expander("🏷️ Edit Dish Name", expanded=True):
        current_name = analysis.dish_name
        new_name = st.text_input(
            "Dish Name", value=current_name, help="Enter a new name for this dish"
        )

        if new_name != current_name:
            modifications["dish_name"] = new_name
            st.success(f"✅ Dish name will be changed to: '{new_name}'")

    # Ingredient modifications
    with st.expander("🥕 Modify Ingredients", expanded=True):
        # Existing ingredients modification
        st.write("#### Existing Ingredients")
        ingredient_changes = {}
        ingredients_to_remove = []

        for i, ingredient in enumerate(analysis.ingredients):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

            with col1:
                st.write(f"**{ingredient.ingredient}**")
                st.write(
                    f"Current: {ingredient.weight:.0f}g, {ingredient.calories} cal"
                )

            with col2:
                # Weight modification
                new_weight = st.number_input(
                    "Weight (g)",
                    min_value=0.0,
                    value=float(ingredient.weight),
                    step=5.0,
                    key=f"weight_{i}",
                    help=f"Current weight: {ingredient.weight:.0f}g",
                )

                if abs(new_weight - ingredient.weight) > 0.1:  # Changed significantly
                    ingredient_changes[ingredient.ingredient] = new_weight
                    # Calculate estimated calorie change
                    cal_ratio = (
                        new_weight / ingredient.weight if ingredient.weight > 0 else 0
                    )
                    est_new_calories = int(ingredient.calories * cal_ratio)
                    st.write(f"→ ~{est_new_calories} cal")

            with col3:
                # Show current nutrition info
                st.write(f"P: {ingredient.protein:.1f}g")
                st.write(f"C: {ingredient.carbohydrates:.1f}g")
                st.write(f"F: {ingredient.fat:.1f}g")

            with col4:
                # Remove option
                if st.checkbox("Remove", key=f"remove_{i}"):
                    ingredients_to_remove.append(ingredient.ingredient)
                    st.warning("Will remove")

        # New ingredients addition
        st.write("#### Add New Ingredients")

        # Initialize ingredients to add in session state
        if "new_ingredients" not in st.session_state:
            st.session_state.new_ingredients = []

        # Initialize new ingredient input key
        if "new_ingredient_input_key" not in st.session_state:
            st.session_state.new_ingredient_input_key = 0

        col1, col2 = st.columns([3, 1])
        with col1:
            new_ingredient_name = st.text_input(
                "Ingredient name",
                placeholder="e.g., Extra avocado, Hot sauce, Cheese",
                help="Enter the name of an ingredient to add",
                key=f"new_ingredient_input_{st.session_state.new_ingredient_input_key}",
            )
        with col2:
            if st.button("➕ Add Ingredient") and new_ingredient_name.strip():
                if new_ingredient_name.strip() not in st.session_state.new_ingredients:
                    st.session_state.new_ingredients.append(new_ingredient_name.strip())
                    # Clear the input by changing the key
                    st.session_state.new_ingredient_input_key += 1
                    st.success(f"Added: {new_ingredient_name}")
                    st.rerun()
                else:
                    st.warning("Ingredient already added")

        # Show added ingredients with weight inputs
        if st.session_state.new_ingredients:
            st.write("**Ingredients to add:**")

            # Initialize new ingredient weights in session state
            if "new_ingredient_weights" not in st.session_state:
                st.session_state.new_ingredient_weights = {}

            ingredients_to_remove = []
            for j, new_ing in enumerate(st.session_state.new_ingredients):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"• **{new_ing}**")
                with col2:
                    # Weight input for new ingredient
                    weight_key = f"new_weight_{new_ing}"
                    current_weight = st.session_state.new_ingredient_weights.get(
                        new_ing, 100.0
                    )
                    new_weight = st.number_input(
                        "Weight (g)",
                        min_value=1.0,
                        max_value=1000.0,
                        value=float(current_weight),
                        step=5.0,
                        key=weight_key,
                        help=f"Enter weight for {new_ing}",
                    )
                    st.session_state.new_ingredient_weights[new_ing] = new_weight
                with col3:
                    if st.button("❌", key=f"remove_new_{j}", help=f"Remove {new_ing}"):
                        ingredients_to_remove.append(new_ing)

            # Remove ingredients marked for removal
            for ing_to_remove in ingredients_to_remove:
                st.session_state.new_ingredients.remove(ing_to_remove)
                if ing_to_remove in st.session_state.new_ingredient_weights:
                    del st.session_state.new_ingredient_weights[ing_to_remove]
                st.rerun()

        # Store modifications
        if ingredient_changes:
            modifications["ingredient_weight_changes"] = ingredient_changes
        if ingredients_to_remove:
            modifications["ingredients_to_remove"] = ingredients_to_remove
        if st.session_state.new_ingredients:
            # Create dictionary of ingredient names to weights
            ingredients_with_weights = {}
            for ingredient in st.session_state.new_ingredients:
                weight = st.session_state.new_ingredient_weights.get(ingredient, 100.0)
                ingredients_with_weights[ingredient] = weight
            modifications["ingredients_to_add"] = ingredients_with_weights

    # Portion size modification
    with st.expander("📏 Edit Portion Size"):
        st.info(
            "💡 **Note:** This only changes the portion description text. To scale nutrition values proportionally, manually adjust individual ingredient weights above."
        )

        current_portion = analysis.portion_size
        new_portion = st.text_input(
            "Portion Size",
            value=current_portion,
            help="Describe the portion size (e.g., '1 large serving', '2 cups')",
        )

        if new_portion != current_portion:
            modifications["portion_size"] = new_portion
            st.success(f"✅ Portion size will be changed to: '{new_portion}'")

    # Show modification summary
    if modifications:
        render_modification_summary(modifications, analysis)

    # Create and return UserModification object
    if modifications:
        return UserModification(**modifications)
    else:
        st.info("💡 No modifications made. The analysis will remain unchanged.")
        return None


def render_modification_summary(modifications: Dict, original_analysis):
    """Render a summary of modifications to be applied."""
    st.markdown("### 📋 Modification Summary")

    changes_made = []

    if "dish_name" in modifications:
        changes_made.append(
            f"🏷️ **Dish name:** {original_analysis.dish_name} → {modifications['dish_name']}"
        )

    if "ingredient_weight_changes" in modifications:
        changes_made.append("⚖️ **Weight changes:**")
        for ingredient, new_weight in modifications[
            "ingredient_weight_changes"
        ].items():
            # Find original weight
            original_weight = next(
                (
                    ing.weight
                    for ing in original_analysis.ingredients
                    if ing.ingredient == ingredient
                ),
                0,
            )
            change = new_weight - original_weight
            changes_made.append(
                f"   • {ingredient}: {original_weight:.0f}g → {new_weight:.0f}g ({change:+.0f}g)"
            )

    if "ingredients_to_remove" in modifications:
        changes_made.append("❌ **Remove ingredients:**")
        for ingredient in modifications["ingredients_to_remove"]:
            changes_made.append(f"   • {ingredient}")

    if "ingredients_to_add" in modifications:
        changes_made.append("➕ **Add ingredients:**")
        for ingredient, weight in modifications["ingredients_to_add"].items():
            changes_made.append(f"   • {ingredient} ({weight:.0f}g)")

    if "portion_size" in modifications:
        changes_made.append(
            f"📏 **Portion size:** {original_analysis.portion_size} → {modifications['portion_size']}"
        )

    # Display changes
    if changes_made:
        st.success("**Changes to be applied:**")
        for change in changes_made:
            st.markdown(change)

        # Estimate impact
        estimate_modification_impact(modifications, original_analysis)
    else:
        st.info("No modifications to apply.")


def estimate_modification_impact(modifications: Dict, original_analysis):
    """Estimate the nutritional impact of modifications."""
    st.markdown("#### 📊 Estimated Impact")

    estimated_calorie_change = 0
    impact_notes = []

    # Estimate from weight changes
    if "ingredient_weight_changes" in modifications:
        for ingredient, new_weight in modifications[
            "ingredient_weight_changes"
        ].items():
            # Find the ingredient in original analysis
            original_ing = next(
                (
                    ing
                    for ing in original_analysis.ingredients
                    if ing.ingredient == ingredient
                ),
                None,
            )
            if original_ing:
                if original_ing.weight > 0:
                    ratio = new_weight / original_ing.weight
                    calorie_change = original_ing.calories * (ratio - 1)
                    estimated_calorie_change += calorie_change

                    if abs(calorie_change) > 10:  # Only show significant changes
                        impact_notes.append(
                            f"• {ingredient}: {calorie_change:+.0f} calories"
                        )

    # Estimate from removed ingredients
    if "ingredients_to_remove" in modifications:
        for ingredient_name in modifications["ingredients_to_remove"]:
            removed_ing = next(
                (
                    ing
                    for ing in original_analysis.ingredients
                    if ing.ingredient == ingredient_name
                ),
                None,
            )
            if removed_ing:
                estimated_calorie_change -= removed_ing.calories
                impact_notes.append(
                    f"• Remove {ingredient_name}: -{removed_ing.calories} calories"
                )

    # Note about added ingredients
    if "ingredients_to_add" in modifications:
        total_added_weight = sum(modifications["ingredients_to_add"].values())
        impact_notes.append(
            f"• Added {len(modifications['ingredients_to_add'])} ingredients ({total_added_weight:.0f}g total) - calories will be calculated by AI"
        )

    # Display impact
    if estimated_calorie_change != 0:
        st.metric(
            "Estimated Calorie Change",
            f"{estimated_calorie_change:+.0f}",
            help="This is an estimate. Final values will be calculated by AI.",
        )

    if impact_notes:
        st.write("**Impact breakdown:**")
        for note in impact_notes:
            st.write(note)

    st.info(
        "🤖 **Note:** Final nutrition values will be recalculated by our AI system based on your modifications."
    )


def clear_modification_state():
    """Clear modification-related session state."""
    keys_to_clear = [
        key
        for key in st.session_state.keys()
        if key.startswith(
            (
                "new_ingredients",
                "new_ingredient_weights",
                "weight_",
                "remove_",
                "new_weight_",
            )
        )
    ]
    for key in keys_to_clear:
        del st.session_state[key]


def render_quick_modifications(analysis):
    """Render quick modification buttons for common changes."""
    st.markdown("#### ⚡ Quick Modifications")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🥑 Add Avocado"):
            if "new_ingredients" not in st.session_state:
                st.session_state.new_ingredients = []
            if "new_ingredient_weights" not in st.session_state:
                st.session_state.new_ingredient_weights = {}
            if "Avocado" not in st.session_state.new_ingredients:
                st.session_state.new_ingredients.append("Avocado")
                st.session_state.new_ingredient_weights["Avocado"] = (
                    50.0  # Default avocado weight
                )
                st.rerun()

    with col2:
        if st.button("🧀 Add Cheese"):
            if "new_ingredients" not in st.session_state:
                st.session_state.new_ingredients = []
            if "new_ingredient_weights" not in st.session_state:
                st.session_state.new_ingredient_weights = {}
            if "Cheese" not in st.session_state.new_ingredients:
                st.session_state.new_ingredients.append("Cheese")
                st.session_state.new_ingredient_weights["Cheese"] = (
                    30.0  # Default cheese weight
                )
                st.rerun()

    with col3:
        if st.button("🌶️ Add Hot Sauce"):
            if "new_ingredients" not in st.session_state:
                st.session_state.new_ingredients = []
            if "new_ingredient_weights" not in st.session_state:
                st.session_state.new_ingredient_weights = {}
            if "Hot Sauce" not in st.session_state.new_ingredients:
                st.session_state.new_ingredients.append("Hot Sauce")
                st.session_state.new_ingredient_weights["Hot Sauce"] = (
                    10.0  # Default hot sauce weight
                )
                st.rerun()

    with col4:
        if st.button("🥗 Make it Salad"):
            # Suggest reducing portions and adding vegetables
            st.info(
                "💡 Consider reducing portions and adding lettuce, tomatoes, and cucumber!"
            )


def validate_modifications(modifications: Dict) -> tuple[bool, List[str]]:
    """Validate the modifications before applying."""
    errors = []

    # Check for empty dish name
    if "dish_name" in modifications and not modifications["dish_name"].strip():
        errors.append("Dish name cannot be empty")

    # Check weight changes are positive
    if "ingredient_weight_changes" in modifications:
        for ingredient, weight in modifications["ingredient_weight_changes"].items():
            if weight < 0:
                errors.append(f"Weight for {ingredient} cannot be negative")
            elif weight > 1000:  # Reasonable upper limit
                errors.append(f"Weight for {ingredient} seems too high (>1000g)")

    # Check for duplicate ingredients in add list and validate weights
    if "ingredients_to_add" in modifications:
        ingredients = modifications["ingredients_to_add"]
        for ingredient, weight in ingredients.items():
            if weight <= 0:
                errors.append(
                    f"Weight for new ingredient '{ingredient}' must be positive"
                )
            elif weight > 1000:
                errors.append(
                    f"Weight for new ingredient '{ingredient}' seems too high (>1000g)"
                )

    return len(errors) == 0, errors
