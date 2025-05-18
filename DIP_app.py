import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(page_title="MedVision: Medical Image Analyzer", layout="wide")
st.title("🧠 MedVision - Medical Image Analyzer")

# File uploader to upload medical image
uploaded_file = st.file_uploader("Upload a medical image (X-ray or MRI)", type=["jpg", "jpeg", "png"])

# If file is uploaded, proceed with processing
if uploaded_file:
    # Convert uploaded file to NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale

    # Sidebar for user options
    st.sidebar.title("🔧 Processing Options")

    # ========== FILTERING ==========
    # Filter type selection
    filter_type = st.sidebar.selectbox("Select Filter", ["None", "Gaussian Blur", "Median Filter"])
    filtered_image = image.copy()  # Start with original image

    # Apply Gaussian Blur
    if filter_type == "Gaussian Blur":
        ksize = st.sidebar.slider("Kernel Size (Gaussian)", 3, 15, 5, step=2)
        filtered_image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # Apply Median Filter
    elif filter_type == "Median Filter":
        ksize = st.sidebar.slider("Kernel Size (Median)", 3, 15, 5, step=2)
        filtered_image = cv2.medianBlur(image, ksize)

    # ========== CLAHE (Contrast Limited Adaptive Histogram Equalization) ==========
    # Enhance contrast if selected
    clahe = None
    if st.sidebar.checkbox("Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"):
        clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 10.0, 2.0)
        tile_grid_size = st.sidebar.slider("Tile Grid Size", 8, 32, 8, step=8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        filtered_image = clahe.apply(filtered_image)

    # ========== SEGMENTATION ==========
    # Select segmentation method
    seg_type = st.sidebar.selectbox("Segmentation Method", ["None", "Otsu Thresholding", "Adaptive Thresholding"])
    segmented = None

    # Otsu's Thresholding
    if seg_type == "Otsu Thresholding":
        _, segmented = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive Thresholding
    elif seg_type == "Adaptive Thresholding":
        block_size = st.sidebar.slider("Block Size", 3, 25, 11, step=2)
        C = st.sidebar.slider("Constant C", 1, 10, 2)
        segmented = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block_size, C)

    # ========== EDGE DETECTION ==========
    # Canny Edge Detection toggle
    detect_edges = st.sidebar.checkbox("Apply Canny Edge Detection")
    edges = None
    if detect_edges:
        t1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
        t2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)
        edges = cv2.Canny(filtered_image, t1, t2)

    # ========== FRACTURE DETECTION PARAMETERS ==========
    # User-defined contour filtering parameters
    min_area = st.sidebar.slider("Min Contour Area for Fracture", 100, 3000, 1000, step=100)
    circularity_threshold = st.sidebar.slider("Circularity Threshold", 0.1, 1.0, 0.4, step=0.05)

    # ========== FRACTURE DETECTION USING CONTOURS ==========
    show_contours = st.sidebar.checkbox("Show Contours")
    contour_result = cv2.cvtColor(filtered_image.copy(), cv2.COLOR_GRAY2BGR)  # Convert to color image for drawing
    fracture_flag = False  # Flag to determine fracture presence
    fracture_boxes = []  # Store bounding boxes for fractures

    # Statistics for contour-based diagnosis
    total_contours = small_fragments = irregular_shapes = 0

    if segmented is not None:
        # Find external contours
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_contours = len(contours)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Identify small fragments
            if area < min_area:
                small_fragments += 1
                x, y, w, h = cv2.boundingRect(cnt)
                fracture_boxes.append((x, y, w, h))

            # Identify irregular shapes
            if circularity < circularity_threshold:
                irregular_shapes += 1

        # Set fracture flag based on conditions
        if total_contours > 5 or small_fragments > 5 or irregular_shapes > 3:
            fracture_flag = True

        # Draw contours and bounding boxes
        if show_contours:
            cv2.drawContours(contour_result, contours, -1, (0, 255, 0), 1)
            for (x, y, w, h) in fracture_boxes:
                cv2.rectangle(contour_result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # ========== DISPLAY RESULTS ==========
    st.subheader("📊 Results")

    # First row of images
    row1 = st.columns(4)
    row1[0].image(image, caption="Original", use_container_width=True, channels="GRAY")
    row1[1].image(filtered_image, caption=f"Filtered ({filter_type}) + CLAHE", use_container_width=True, channels="GRAY")
    if segmented is not None:
        row1[2].image(segmented, caption=f"Segmented ({seg_type})", use_container_width=True, channels="GRAY")
    else:
        row1[2].empty()
    row1[3].empty()

    # Second row of images
    row2 = st.columns(4)
    if detect_edges:
        row2[0].image(edges, caption="Canny Edges", use_container_width=True, channels="GRAY")
    else:
        row2[0].empty()

    if show_contours:
        row2[1].image(contour_result, caption="Contours & Fractures", use_container_width=True)
    else:
        row2[1].empty()

    row2[2].empty()
    row2[3].empty()

    # ========== DIAGNOSIS SUMMARY ==========
    st.subheader("🩻 Diagnosis Summary")
    st.write(f"Total Contours: {total_contours}")
    st.write(f"Small Fragments: {small_fragments}")
    st.write(f"Irregular Shapes: {irregular_shapes}")

    # Display diagnosis result
    if fracture_flag:
        st.error("🚨 Fracture Likely Detected Based on Contour Analysis")
    else:
        st.success("✅ No significant fracture indicators found")

    # Optional: Show analysis parameters
    if show_contours:
        st.subheader("🛠 Contour Analysis Parameters")
        st.write(f"Min Contour Area for Fracture: {min_area}")
        st.write(f"Circularity Threshold: {circularity_threshold}")
        st.write(f"Total Contours Detected: {total_contours}")
        st.write(f"Small Fragments: {small_fragments}")
        st.write(f"Irregular Shapes: {irregular_shapes}")

    # ========== DOWNLOAD SECTION ==========
    st.markdown("---")
    st.subheader("📥 Download Image")

    # Dictionary of processed image outputs
    options = {
        "Segmented Image": segmented,
        "Filtered Image": filtered_image,
        "Canny Edge Detection": edges,
        "Contours with Highlights": contour_result if show_contours else None
    }

    # List available outputs
    available = [k for k, v in options.items() if v is not None]
    selection = st.selectbox("Choose Image to Download", available)

    # Create and download selected image
    if selection:
        img = options[selection]
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        byte_img = buffer.getvalue()

        st.download_button(f"Download {selection}", data=byte_img,
                           file_name=f"{selection.lower().replace(' ', '_')}.png", mime="image/png")

    # Footer
    st.markdown("---")
    st.markdown("Built by **Affan & Zaryab** for the **DIP Project** - Medical Image Processing using classical techniques.")
