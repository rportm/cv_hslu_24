
import matplotlib.pyplot as plt

def plot_training_history(history):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot training history
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot dice coefficients
    ax2.plot(history.history['dice_coefficient'], label='Training Dice Coefficient')
    ax2.plot(history.history['val_dice_coefficient'], label='Validation Dice Coefficient')
    ax2.set_title('Dice Coefficient')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()