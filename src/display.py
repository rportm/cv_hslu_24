import nibabel as nib
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom
import numpy as np


# Display images with mask overlay
def display_image(image, mask=None, alpha=0.7, rows=3):
    num_slices = image.shape[2]
    cols = int(np.ceil(num_slices / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    for i in range(num_slices):
        axes[i].imshow(image[:, :, i], cmap='gray')

        if mask is not None:
            overlay = np.zeros((*mask[:, :, i].shape, 4))
            overlay[mask[:, :, i] > 0] = [1, 0, 0, alpha]  # Red color with alpha transparency

            axes[i].imshow(overlay)

        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def display_slices(slices, masks=None, predictions=None, alpha=0.7, rows=3):
    num_slices = slices.shape[0]
    cols = int(np.ceil(num_slices / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    for i in range(num_slices):
        axes[i].imshow(slices[i, :, :], cmap='gray')

        # Initialize a combined overlay
        combined_overlay = np.zeros((*slices[i].shape, 4))

        if masks is not None and predictions is not None:
            # Set yellow color where both overlap
            combined_overlay[np.logical_and(masks[i, :, :] > 0, predictions[i, :, :] > 0)] = [1, 1, 0, alpha]

            # Set red color where only mask is present
            combined_overlay[np.logical_and(masks[i, :, :] > 0, predictions[i, :, :] == 0)] = [1, 0, 0, alpha]

            # Set blue color where only prediction is present
            combined_overlay[np.logical_and(masks[i, :, :] == 0, predictions[i, :, :] > 0)] = [0, 0, 1, alpha]

        elif masks is not None:
            combined_overlay[masks[i, :, :] > 0] = [1, 0, 0, alpha]  # Red color with alpha transparency

        elif predictions is not None:
            combined_overlay[predictions[i, :, :] > 0] = [0, 1, 0, alpha]  # Green color with alpha transparency

        # Display the combined overlay
        axes[i].imshow(np.squeeze(combined_overlay))

        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()




