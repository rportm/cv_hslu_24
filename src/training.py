import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

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

def combined_loss(y_true, y_pred, alpha=0.5):
    return alpha * dice_loss(y_true, y_pred) + (1 - alpha) * tf.keras.losses.binary_crossentropy(y_true, y_pred)


# Define the U-Net model with ResNet50 backbone
def build_unet_resnet50(input_shape=(256, 256, 1)):
    inputs = layers.Input(input_shape)

    # Expand input to 3 channels using concatenation instead of Conv2D
    x = layers.Concatenate()([inputs, inputs, inputs])

    # ResNet50 backbone with custom input shape
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

    # # Freeze ResNet50 weights
    # for layer in resnet.layers:
    #     layer.trainable = False

    # Extract layers from ResNet50
    s1 = resnet.get_layer('conv1_relu').output
    s2 = resnet.get_layer('conv2_block3_out').output
    s3 = resnet.get_layer('conv3_block4_out').output
    s4 = resnet.get_layer('conv4_block6_out').output

    # Bridge
    b1 = resnet.get_layer('conv5_block3_out').output

    # Create the model with custom input
    resnet_model = tf.keras.Model(inputs=resnet.inputs, outputs=[s1, s2, s3, s4, b1])

    # Use the custom input in the resnet model
    s1, s2, s3, s4, b1 = resnet_model(x)

    # Decoder (upsampling)
    d1 = layers.UpSampling2D((2, 2))(b1)
    d1 = layers.concatenate([d1, s4])
    d1 = layers.Conv2D(512, 3, activation='relu', padding='same')(d1)
    d1 = layers.Conv2D(512, 3, activation='relu', padding='same')(d1)

    d2 = layers.UpSampling2D((2, 2))(d1)
    d2 = layers.concatenate([d2, s3])
    d2 = layers.Conv2D(256, 3, activation='relu', padding='same')(d2)
    d2 = layers.Conv2D(256, 3, activation='relu', padding='same')(d2)

    d3 = layers.UpSampling2D((2, 2))(d2)
    d3 = layers.concatenate([d3, s2])
    d3 = layers.Conv2D(128, 3, activation='relu', padding='same')(d3)
    d3 = layers.Conv2D(128, 3, activation='relu', padding='same')(d3)

    d4 = layers.UpSampling2D((2, 2))(d3)
    d4 = layers.concatenate([d4, s1])
    d4 = layers.Conv2D(64, 3, activation='relu', padding='same')(d4)
    d4 = layers.Conv2D(64, 3, activation='relu', padding='same')(d4)

    d5 = layers.UpSampling2D(size=(2, 2))(d4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d5)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model