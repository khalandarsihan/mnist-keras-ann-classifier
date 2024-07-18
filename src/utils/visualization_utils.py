# src/utils/visualization_utils.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def visualize_sample_images_with_labels(x, y, title):
    num_samples = 16
    indices = np.random.choice(len(x), num_samples, replace=False)
    x_samples, y_samples = x[indices], y[indices]
    
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for ax, image, label in zip(axes.flatten(), x_samples, y_samples):
        ax.imshow(image, cmap='gray')
        ax.set_title(f'label: {label}')
        ax.axis('off')
    
    plt.suptitle(f'{title} - Sample Images with Labels')
    plt.tight_layout()
    plt.show()

def visualize_image_mean_and_std(x, title):
    """
From the visualization of the mean and standard deviation images, we can infer the following:

1. **Mean Image**:
   - The mean image represents the average pixel intensity values across all the images in the training set.
   - This image appears blurry but shows the general shape of the digits. This is because averaging many images of digits will highlight common structures (e.g., the circular shape of '0' or vertical lines in '1') while smoothing out individual variations.
   - The central region is brighter, indicating higher average pixel intensities, which makes sense since the digits are generally centered in the images.

2. **Standard Deviation Image**:
   - The standard deviation image shows the variation or spread of pixel intensity values around the mean for each pixel position.
   - Brighter regions in this image indicate higher variability in those pixel positions. These are areas where the digits have more variation in shape, size, or position.
   - Darker regions have lower variability, indicating more consistent pixel values across different images. These are often the background regions or parts of the image that don't change much between different samples.

Overall, these images provide a good overview of the dataset's characteristics:
- The mean image gives an idea of the average appearance of the digits.
- The standard deviation image highlights which parts of the image have more or less variability, giving insights into the consistency of the digit shapes and their positioning in the images.

This information can be useful for understanding the dataset and for making decisions about preprocessing steps, such as normalization and data augmentation, which might be necessary for improving model performance.
"""
    mean_image = np.mean(x, axis=0)
    std_image = np.std(x, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(mean_image, cmap='gray')
    axes[0].set_title('Mean Image')
    axes[0].axis('off')
    
    axes[1].imshow(std_image, cmap='gray')
    axes[1].set_title('Standard Deviation Image')
    axes[1].axis('off')
    
    fig.suptitle(f'{title} - Image Mean and Standard Deviation')
    plt.tight_layout()
    plt.show()

def visualize_heatmap_of_first_image(x, y, title):
    """
The heatmap visualization of the first image in the MNIST dataset provides several insights:

1. **Pixel Intensity Representation**:
   - The heatmap shows the intensity of each pixel in the image, where the values range from 0 (black) to 255 (white). This range is typical for grayscale images.
   - In this specific image, which is labeled as a '5', we can see that the shape of the digit is clearly visible. The white pixels represent the parts of the image where the digit is present, while the black pixels represent the background.

2. **Digit Structure**:
   - The heatmap allows us to see the structure and stroke of the digit '5'. The brightness of each pixel indicates its intensity, showing how the digit is formed with varying thickness and curves.

3. **Detailed Analysis**:
   - The color bar on the right side of the heatmap provides a reference for the pixel intensity values. This can be useful for understanding the variation in intensity within the digit itself.
   - The heatmap can highlight any anomalies or noise present in the digit. For instance, if there are unexpected bright spots in the background or inconsistencies in the digit’s stroke, they will be visible in the heatmap.

4. **Data Characteristics**:
   - This visualization helps in understanding the quality and characteristics of the data. It shows that the digit is well-centered and clearly distinguishable, which is a common feature of the MNIST dataset.
   - By examining multiple heatmaps, you can get a sense of the variability in how different digits are written, which can inform preprocessing steps or augmentations needed for your model.

In summary, the heatmap provides a detailed view of the pixel intensities of the first image, allowing for a closer examination of the digit's structure and any potential issues with the data. This kind of visualization is useful for gaining a deeper understanding of individual samples in your dataset.
"""
    plt.figure(figsize=(6, 6))
    sns.heatmap(x[0], cmap='gray', annot=False, cbar=True)
    plt.title(f'Heatmap of First Image (Label: {y[0]})')
    plt.axis('off')
    plt.show()

def visualize_class_distribution(y_train, y_test, title):
    def class_dist(y):
        labels = [label for label in y]
        return pd.Series(labels).value_counts()

    train_dist = class_dist(y_train)
    test_dist = class_dist(y_test)

    df = pd.DataFrame({
        'Train_dist': train_dist,
        'Test_dist': test_dist
    })

    df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'{title} - Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.show()
