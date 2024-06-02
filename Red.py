from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.write("""
# Simple Image Dimensionality Reduction App

Insert the URL of an image and see the results of applying Singular Value Decomposition (SVD) to reduce its dimensionality!         
""")

# Input field for the image URL
url = st.text_input("Enter the URL of the image:")

def svd_reduction(image, top_k):
    if len(image.shape) == 3:
        channels = []
        for i in range(3):
            U, s, Vt = np.linalg.svd(image[:, :, i], full_matrices=False)
            S = np.diag(s[:top_k])
            reduced_channel = U[:, :top_k] @ S @ Vt[:top_k, :]
            channels.append(reduced_channel)
        reduced_image = np.stack(channels, axis=2)
    else:
        U, s, Vt = np.linalg.svd(image, full_matrices=False)
        S = np.diag(s[:top_k])
        reduced_image = U[:, :top_k] @ S @ Vt[:top_k, :]
    return reduced_image

if url:
    try:
        # Read the image from the URL
        image = io.imread(url)

        # Normalize the image data to the range [0, 1]
        if image.dtype != np.float32:
            image = image / 255.0

        # Display the original image
        st.image(image, caption='Original Image', use_column_width=True)

        # Set top_k to reduce the dimensionality
        top_k = 31

        # Reconstruct the image with reduced dimensionality
        reconstructed_image = svd_reduction(image, top_k)

        # Clip the values to the range [0, 1] to avoid display issues
        reconstructed_image = np.clip(reconstructed_image, 0, 1)

        # Display the reconstructed image
        st.image(reconstructed_image, caption=f'Reconstructed Image with top {top_k} singular values', use_column_width=True)

    except Exception as e:
        st.write("Error loading image. Please check the URL and try again.")
        st.write(f"Error details: {e}")
