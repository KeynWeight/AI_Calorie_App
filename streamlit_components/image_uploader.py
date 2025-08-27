# streamlit_components/image_uploader.py
import streamlit as st
import os


def render_image_uploader():
    """Render the simple image upload interface."""

    st.markdown(
        """
    <div style="text-align: center; padding: 20px; border: 2px dashed #ccc; border-radius: 10px; margin: 20px 0;">
        <h3>📸 Upload your food image</h3>
        <p>Use the file uploader below to select an image</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.write("### Upload from device")
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=["png", "jpg", "jpeg"],
        help="Upload an image of your food for AI analysis",
    )

    if uploaded_file:
        # Display uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                uploaded_file,
                caption=f"Uploaded: {uploaded_file.name}",
                width="stretch",
            )

            # Image info
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**File type:** {uploaded_file.type}")

    return uploaded_file


def render_image_preview(image_path: str):
    """Render a preview of the current image being analyzed."""
    if not image_path or not os.path.exists(image_path):
        return

    st.subheader("🖼️ Current Image")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(image_path, caption="Image being analyzed", width="stretch")

    with col2:
        # Image metadata
        try:
            from PIL import Image

            img = Image.open(image_path)

            st.write("**Image Details:**")
            st.write(f"• Size: {img.size[0]} × {img.size[1]} pixels")
            st.write(f"• Format: {img.format}")
            st.write(f"• Mode: {img.mode}")

            # File size
            file_size = os.path.getsize(image_path)
            st.write(f"• File size: {file_size / 1024:.1f} KB")

        except Exception:
            st.write("Could not load image metadata")


def validate_image(uploaded_file):
    """Validate uploaded image file."""
    if not uploaded_file:
        return False, "No file uploaded"

    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        return (
            False,
            f"File too large: {uploaded_file.size / 1024 / 1024:.1f}MB (max 10MB)",
        )

    # Check file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if uploaded_file.type not in allowed_types:
        return False, f"Unsupported file type: {uploaded_file.type}"

    # Try to open with PIL to verify it's a valid image
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(uploaded_file.getvalue()))

        # Check image dimensions (max 2048x2048)
        max_dim = 2048
        if img.size[0] > max_dim or img.size[1] > max_dim:
            return (
                False,
                f"Image too large: {img.size[0]}×{img.size[1]} (max {max_dim}×{max_dim})",
            )

        return True, "Valid image"

    except Exception as e:
        return False, f"Invalid image file: {str(e)}"
