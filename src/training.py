import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.regularizers import l2
import keras_tuner as kt
import numpy as np
import math
from scipy.ndimage import gaussian_filter


def get_augment_data_generator(mask: bool):
    # Create a local random number generator
    rng = np.random.RandomState(42)

    def brightness_range(img, low=0.8, high=1.2):
        return np.clip(img * rng.uniform(low, high), 0, 1)

    def add_gaussian_noise(img, mean=0, std=0.01):
        noise = rng.normal(mean, std, img.shape)
        return np.clip(img + noise, 0, 1)

    def apply_gaussian_blur(img, sigma_range=(0.1, 0.5)):
        sigma = rng.uniform(*sigma_range)
        return gaussian_filter(img, sigma=sigma)

    return ImageDataGenerator(
        rotation_range=15,  # Increased rotation range
        width_shift_range=0.08,  # Increased shift range
        height_shift_range=0.08,
        zoom_range=0.1,  # Increased zoom range
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant',  # Changed to 'constant' to better handle border areas
        cval=0,  # Fill value for areas outside the input boundaries
        shear_range=5,  # Added shear transformation
        preprocessing_function=lambda x: (
            brightness_range(
                add_gaussian_noise(
                    apply_gaussian_blur(x)
                )
            ) if not mask else x
        )
    )

# Custom generator function
def get_combined_iterator(slice_generator, mask_generator):
    train_generator = zip(slice_generator, mask_generator)
    for (img, mask) in train_generator:
        yield img, mask

# Get augmented train and validation iterators
def get_augment_iterators(slices_train, masks_train, slices_val, masks_val, seed=1, batch_size=32):
    # Flow from memory
    slice_iterator_train = get_augment_data_generator(False).flow(slices_train, seed=seed, batch_size=batch_size)
    mask_iterator_train = get_augment_data_generator(True).flow(masks_train, seed=seed, batch_size=batch_size)

    # Combined generator
    iterator_train = get_combined_iterator(slice_iterator_train, mask_iterator_train)

    # Validation iterator
    slice_iterator_val = ImageDataGenerator().flow(slices_val, seed=seed, batch_size=batch_size)
    mask_iterator_val = ImageDataGenerator().flow(masks_val, seed=seed, batch_size=batch_size)

    iterator_val = get_combined_iterator(slice_iterator_val, mask_iterator_val)

    return iterator_train, iterator_val


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
def build_unet_resnet50(l2_reg=0.01, dropout=0.3, input_shape=(256, 256, 1)):
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
    d1 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(d1)
    d1 = layers.Dropout(dropout)(d1)  # Add after Conv2D layers

    d2 = layers.UpSampling2D((2, 2))(d1)
    d2 = layers.concatenate([d2, s3])
    d2 = layers.Conv2D(256, 3, activation='relu', padding='same')(d2)
    d2 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(d2)
    d2 = layers.Dropout(dropout)(d2)

    d3 = layers.UpSampling2D((2, 2))(d2)
    d3 = layers.concatenate([d3, s2])
    d3 = layers.Conv2D(128, 3, activation='relu', padding='same')(d3)
    d3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(d3)
    d3 = layers.Dropout(dropout)(d3)

    d4 = layers.UpSampling2D((2, 2))(d3)
    d4 = layers.concatenate([d4, s1])
    d4 = layers.Conv2D(64, 3, activation='relu', padding='same')(d4)
    d4 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(d4)
    d4 = layers.Dropout(dropout)(d4)

    d5 = layers.UpSampling2D(size=(2, 2))(d4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d5)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def cosine_annealing(epoch, total_epochs, initial_lr, min_lr):
    return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2


def train_model(model, iterator_train, iterator_val, steps_per_epoch, validation_steps,
                epochs=20, learning_rate=1e-4, use_lr_scheduler=False, best_model_file='best_model.keras'):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=combined_loss,
                  metrics=[dice_coefficient])

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_model_file, save_best_only=True,
                                           monitor='val_dice_coefficient', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_dice_coefficient', patience=10, mode='max',
                                         restore_best_weights=True)
    ]

    if use_lr_scheduler:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: cosine_annealing(epoch, total_epochs=epochs, initial_lr=learning_rate, min_lr=1e-6)
        )
        callbacks.append(lr_scheduler)

    # Train the model
    history = model.fit(
        iterator_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=iterator_val,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history


def build_hypermodel(hp):
    # Hyperparameters to tune
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log')

    # Clear the session, try to avoid memory leak: https://github.com/keras-team/keras-tuner/issues/395
    tf.keras.backend.clear_session()

    # Build the model using the hyperparameters
    model = build_unet_resnet50(l2_reg, dropout_rate, input_shape=(256, 256, 1))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coefficient])

    return model


def get_tuner(epochs=50):
    # Initialize the Keras Tuner
    return kt.Hyperband(
        build_hypermodel,
        objective=kt.Objective('val_dice_coefficient', direction='max'),
        max_epochs=epochs,
        factor=3,
        hyperband_iterations=2,
        project_name='hyperparam_tuning'
    )


def hyperparam_tuning(iterator_train, iterator_val, epochs, steps_per_epoch, validation_steps):
    # Initialize the Keras Tuner
    tuner = get_tuner(epochs)

    # Early stopping callback
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coefficient', patience=5, mode='max')

    # TensorBoard callback for visualization
    tensorboard_cb = tf.keras.callbacks.TensorBoard('logs/hparam_tuning')

    # Perform hyperparameter search
    tuner.search(
        iterator_train,
        steps_per_epoch=steps_per_epoch,
        validation_data=iterator_val,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[stop_early, tensorboard_cb]
    )

    return tuner
