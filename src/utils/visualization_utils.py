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
