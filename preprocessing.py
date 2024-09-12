import nibabel as nib
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom
import numpy as np

def load_nii_files(directory):
    nii_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('Flair.nii'):
                nii_files.append(os.path.join(root, file))
    return sorted(nii_files)


def display_all_slices(nii_file_path):
    img = nib.load(nii_file_path)
    data = img.get_fdata()

    # Display all slices in a row
    fig, axes = plt.subplots(1, data.shape[2], figsize=(20, 20))
    for i in range(data.shape[2]):
        axes[i].imshow(data[:, :, i], cmap='gray')
        axes[i].axis('off')
    plt.show()


# Display slices with mask overlay
def display_slices_with_masks(slices, masks=None, alpha=0.7, rows=3):
    num_slices = slices.shape[2]
    cols = int(np.ceil(num_slices / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    for i in range(num_slices):
        # Display the original slice
        axes[i].imshow(slices[:, :, i], cmap='gray')

        # Create a red transparent mask
        if masks is not None:
            red_mask = np.zeros((*masks[:, :, i].shape, 4))
            red_mask[masks[:, :, i] > 0] = [1, 0, 0, alpha]  # Red color with alpha transparency

            # Overlay the red transparent mask
            axes[i].imshow(red_mask)

        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Function to resize a 3D image to the target shape
def resize_volume(img, target_shape):
    """
    Resize a 3D volume to the target shape.
    
    Parameters:
        img (numpy.ndarray): 3D numpy array of the image.
        target_shape (tuple): Desired output shape (x, y, z).
    
    Returns:
        numpy.ndarray: Resized 3D image.
    """
    # Calculate the resize factor for each dimension
    factors = [target_shape[i] / img.shape[i] for i in range(3)]
    
    # Resize using the calculated factors
    resized_img = zoom(img, factors, order=1)  # order=1 for bilinear interpolation
    
    return resized_img


# Load the NIfTI file using nibabel
def load_nii_file(file_path):
    """
    Load a NIfTI file and return the image data as a numpy array.
    
    Parameters:
        file_path (str): Path to the .nii or .nii.gz file.
    
    Returns:
        numpy.ndarray: 3D numpy array of the image data.
    """
    nii_img = nib.load(file_path)
    img_data = nii_img.get_fdata()  # Load the data as a numpy array
    
    return img_data, nii_img.affine  # Also return the affine matrix for saving later

    
def min_max_normalize(image, new_min=0, new_max=1):
    """
    Normalize an image using min-max normalization.
    
    Parameters:
        image (numpy.ndarray): The input image.
        new_min (float): The minimum value of the normalized image. Default is 0.
        new_max (float): The maximum value of the normalized image. Default is 1.
    
    Returns:
        numpy.ndarray: The normalized image.
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    # Avoid division by zero in case img_max == img_min
    if img_max - img_min == 0:
        return np.zeros(image.shape)
    
    # Min-max normalization
    normalized_image = (image - img_min) / (img_max - img_min)
    
    # Scale to the desired range [new_min, new_max]
    normalized_image = normalized_image * (new_max - new_min) + new_min
    
    return normalized_image


# load all the training data
def load_all_data(nii_files_list):
    all_data = []
    for nii_file_path in nii_files_list:
        img_data, affine = load_nii_file(nii_file_path)
        all_data.append(img_data)
    return all_data


# separate masks and images (they are alternating)
def separate_images_masks(all_data):
    images = all_data[::2]
    masks = all_data[1::2]
    return images, masks


# resize all the images
def resize_all_images(images, x=256, y=256):
    resized_images = []
    for image in images:
        resized_image = resize_volume(image, (x, y, image.shape[-1]))
        resized_images.append(resized_image)
    return resized_images


# Function to filter out slices with no lesions and keep non-empty masks
def filter_slices_with_lesions(mri_images, masks):
    filtered_mri_images = []
    filtered_masks = []
    
    for mri_image, mask in zip(mri_images, masks):
        # Filter slices where the mask is not empty (non-zero elements)
        non_empty_slices = [i for i in range(mri_image.shape[2]) if np.any(mask[:, :, i] > 0)]
        
        if non_empty_slices:
            # Stack the non-empty slices for both MRI and mask
            filtered_mri_images.append(np.stack([mri_image[:, :, i] for i in non_empty_slices], axis=-1))
            filtered_masks.append(np.stack([mask[:, :, i] for i in non_empty_slices], axis=-1))
    
    return filtered_mri_images, filtered_masks


# normalize the images
def normalize_images(images):
    normalized_images = []
    for image in images:
        normalized_image = min_max_normalize(image)
        normalized_images.append(normalized_image)
    return normalized_images

