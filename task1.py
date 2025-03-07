import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset_path = "data.csv"  
df = pd.read_csv(dataset_path)

# Print dataset shape
print("Dataset Shape:", df.shape)

# Extract labels and pixel values
labels = df.iloc[:, 0]  # First column contains labels
images = df.iloc[:, 1:].values  # Remaining columns contain pixel values
images = images.reshape(-1, 28, 28)  # Reshape into 28x28 images

# Display a few sample images with labels
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    axes[i].imshow(images[i], cmap="gray")
    axes[i].set_title(f"Label: {labels[i]}")
    axes[i].axis("off")

plt.show()
# Verify grayscale format (single image test)
print("Single image shape:", images[0].shape)
