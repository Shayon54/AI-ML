import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset_path = "data.csv" 
df = pd.read_csv(dataset_path)

# Display basic dataset info
print("Dataset Shape:", df.shape)
print("First 5 Rows:\n", df.head())

# Extract labels and image data
labels = df.iloc[:, 0]  # First column contains labels
images = df.iloc[:, 1:].values  # Remaining columns contain pixel values

# Check class distribution
print("\nClass Distribution:")
print(labels.value_counts())

# Plot class distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=labels, palette="viridis")
plt.title("Class Distribution of Data Set")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.show()

# Reshape image data
images = images.reshape(-1, 28, 28)

# Display sample images from each category
unique_labels = labels.unique()
fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # 2 rows, 5 columns

for i, label in enumerate(unique_labels[:10]):
    sample_index = labels[labels == label].index[0]  # Get index of first occurrence
    axes[i // 5, i % 5].imshow(images[sample_index], cmap="gray")
    axes[i // 5, i % 5].set_title(f"Label: {label}")
    axes[i // 5, i % 5].axis("off")

plt.show()

# Generate summary statistics for pixel values
print("\nSummary Statistics:")
print(df.iloc[:, 1:].describe())
