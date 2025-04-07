import numpy as np
import matplotlib.pyplot as plt


# plot learning curves from a history dataframe generated during training
def plot_learning_curves(history, w = 9):
    # fail gracefully if there is no history
    if history is None or len(history) == 0:
        print("Empty history, cannot plot")
        return

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    epochs = range(len(history))
    loss = [h['loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    loss = np.convolve(np.pad(loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    val_loss = np.convolve(np.pad(val_loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    ax.plot(epochs, loss, color='green', label='Train')
    ax.plot(epochs, val_loss, color='orange', label='Val.')
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.tight_layout()
    plt.show()

# similar to the previous function, but for a NAS loop (also plots the cost)
def plot_learning_curves_nas(history, w = 9):
    # fail gracefully if there is no history
    if history is None or len(history) == 0:
        print("Empty history, cannot plot")
        return

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    epochs = range(len(history))
    loss = [h['loss'] for h in history]
    nas_loss = [h['nas_loss'] for h in history]
    cost = [h['cost'] for h in history]
    loss = np.convolve(np.pad(loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    nas_loss = np.convolve(np.pad(nas_loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    cost = np.convolve(np.pad(cost, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    line1, = ax.plot(epochs, loss, color='green', label='Train Loss')
    line2, = ax.plot(epochs, nas_loss, color='red', label='Tot. NAS Loss')
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    ax2 = ax.twinx()
    line3, = ax2.plot(epochs, cost, color='blue', label='DNN Cost', linestyle='dashed')
    ax2.set_ylabel('Cost')

    # Combine legends
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)

    plt.tight_layout()
    plt.show()

