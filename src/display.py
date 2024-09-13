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
            red_mask = np.zeros((*mask[:, :, i].shape, 4))
            red_mask[mask[:, :, i] > 0] = [1, 0, 0, alpha]  # Red color with alpha transparency

            axes[i].imshow(red_mask)

        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Display numpy slices with mask overlay
def display_slices(slices, masks=None, predictions=None, alpha=0.7, rows=3):
    num_slices = slices.shape[0]
    cols = int(np.ceil(num_slices / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    for i in range(num_slices):
        axes[i].imshow(slices[i, :, :], cmap='gray')

        if masks is not None:
            mask_overlay = np.zeros((*masks[i].shape, 4))
            mask_overlay[masks[i, :, :] > 0] = [1, 0, 0, alpha]  # Red color with alpha transparency

            axes[i].imshow(np.squeeze(mask_overlay))

        if predictions is not None:
            predictions_overlay = np.zeros((*predictions[i].shape, 4))
            predictions_overlay[predictions[i, :, :] > 0] = [0, 1, 0, alpha]  # Green color with alpha transparency

            axes[i].imshow(np.squeeze(predictions_overlay))

        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()




