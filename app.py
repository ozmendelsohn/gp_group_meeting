import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules from the project
from utils import gaussian_process_interactive

# Set up the Streamlit app
st.title("Interactive Gaussian Process Explorer")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
kernel_choice = st.sidebar.selectbox("Kernel", ["RBF", "Matern"])

# Display the selected options
st.write(f"Selected Noise Level: {noise_level}")
st.write(f"Selected Kernel: {kernel_choice}")

# Placeholder for the Gaussian Process plot
st.write("### Gaussian Process Plot")
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
st.pyplot(fig)

# Call the custom function from utils
gaussian_process_interactive(lambda x: np.sin(x), noise_level)
