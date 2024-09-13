import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a data generator with augmentation
def get_augment_data_generator():
    return ImageDataGenerator(
        rotation_range=10,  # Slight rotations to account for patient positioning
        width_shift_range=0.05,  # Small shifts to simulate patient movement
        height_shift_range=0.05,
        zoom_range=0.05,  # Slight zooming to account for scan variability
        horizontal_flip=True,  # Flips can help with generalization
        vertical_flip=False,  # Vertical flips are less common in medical imaging
        fill_mode='nearest',  # Fills in newly created pixels
        #brightness_range=[0.9, 1.1],  # Slight brightness variations
    )

# Custom generator function
def get_combined_iterator(slice_generator, mask_generator):
    train_generator = zip(slice_generator, mask_generator)
    for (img, mask) in train_generator:
        yield img, mask

# Define the Dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Define the Dice loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)