# create a data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a data generator with augmentation
def get_data_generator():
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
def get_train_generator(slice_generator, mask_generator):
    train_generator = zip(slice_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)