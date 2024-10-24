import tensorflow as tf
import keras_tuner as kt
import math

tf.get_logger().setLevel('WARNING')  # Suppress TensorFlow info messages

# Define the hypermodel function for Keras Tuner
def build_model(hp):
    # Hyperparameters to tune
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log')
    
    # Build the model using the hyperparameters
    model = build_unet_resnet50(l2_reg, dropout_rate, input_shape=(256, 256, 1))
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=combined_loss,
                  metrics=[dice_coefficient])
    return model

# Initialize the Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_dice_coefficient', direction='max'),
    max_epochs=50,
    factor=3,
    hyperband_iterations=2,
    directory='my_dir',
    project_name='hyperparam_tuning'
)

# Early stopping callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coefficient', patience=5, mode='max')

# TensorBoard callback for visualization
tensorboard_cb = tf.keras.callbacks.TensorBoard('logs/hparam_tuning')

# Perform hyperparameter search
tuner.search(
    iterator_train,
    steps_per_epoch=len(all_slices_train) // BATCH_SIZE,
    validation_data=iterator_val,
    validation_steps=len(all_slices_val) // BATCH_SIZE,
    epochs=50,
    callbacks=[stop_early, tensorboard_cb]
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


print(f"""
The hyperparameter search is complete.
The optimal learning rate is {best_hps.get('learning_rate')}.
The optimal L2 regularization strength is {best_hps.get('l2_reg')}.
The optimal dropout rate is {best_hps.get('dropout_rate')}.
""")

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

# Define the cosine annealing function
def cosine_annealing(epoch, total_epochs, initial_lr, min_lr):
    return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: cosine_annealing(epoch, total_epochs=50, initial_lr=best_hps.get('learning_rate'), min_lr=1e-6)
)

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model_l2_reg.keras', save_best_only=True, monitor='val_dice_coefficient', mode='max'),
    lr_scheduler,
    tf.keras.callbacks.EarlyStopping(monitor='val_dice_coefficient', patience=10, mode='max', restore_best_weights=True),
    tensorboard_cb
]

# Train the model with the optimal hyperparameters
history = model.fit(
    iterator_train,
    steps_per_epoch=len(all_slices_train) // BATCH_SIZE,
    validation_data=iterator_val,
    validation_steps=len(all_slices_val) // BATCH_SIZE,
    epochs=50,
    callbacks=callbacks
)

# Evaluate the model
evaluation = model.evaluate(iterator_val, steps=len(all_slices_val) // BATCH_SIZE)
print(f"Validation Loss: {evaluation[0]}")
print(f"Validation Dice Coefficient: {evaluation[1]}")

# Print a summary of the hyperparameter tuning results
tuner.results_summary()