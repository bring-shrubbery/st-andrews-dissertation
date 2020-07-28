import numpy as np
from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt

def plotAndSave(filename, history, model, X_val, y_val):
    # Create the figure.
    f = plt.figure(figsize=(15, 10))
    f.tight_layout()

    # Create subplots for each metric and confusion matrix.
    ax1 = f.add_subplot(231, xlabel='Epoch', ylabel='Accuracy',  ylim=[0, 1])
    ax2 = f.add_subplot(232, xlabel='Epoch', ylabel='MSE',       ylim=[0, 1])
    ax3 = f.add_subplot(233, xlabel='Epoch', ylabel='Precision', ylim=[0, 1])
    ax4 = f.add_subplot(234, xlabel='Epoch', ylabel='Recall',    ylim=[0, 1])
    ax5 = f.add_subplot(235, xlabel='Epoch', ylabel='Loss')
    ax6 = f.add_subplot(236, xlabel='Predicted', ylabel='Actual')

    ax1.plot(history.history['accuracy'], label='accuracy')
    ax1.plot(history.history['val_accuracy'], label='val_accuracy')
    ax1.legend(loc='lower right')

    ax2.plot(history.history['mse'], label='mse')
    ax2.plot(history.history['val_mse'], label='val_mse')
    ax2.legend(loc='lower right')

    ax3.plot(history.history['precision'], label='precision')
    ax3.plot(history.history['val_precision'], label='val_precision')
    ax3.legend(loc='lower right')

    ax4.plot(history.history['recall'], label='recall')
    ax4.plot(history.history['val_recall'], label='val_recall')
    ax4.legend(loc='lower right')

    ax5.plot(history.history['loss'], label='loss')
    ax5.plot(history.history['val_loss'], label='val_loss')
    ax5.set_yscale('log')
    ax5.legend(loc='lower right')

    # Make validation predictions and construct confusion matrix.
    predictions = model.predict(X_val, batch_size=2)
    predictions = [round(p[0]) for p in predictions]
    print("Predictions:", predictions)
    print("Y_val:", y_val)
    cf = confusion_matrix(y_val, predictions)
    cf = np.array(cf)

    # Plot confusion matrix.
    im_id = ax6.imshow(cf, interpolation='nearest')
    ax6.set_xticks(np.arange(2))
    ax6.set_yticks(np.arange(2))
    ax6.set_xticklabels(['positive', 'negative'])
    ax6.set_yticklabels(['positive', 'negative'])
    ax6.set_title("Confusion Matrix")
    f.colorbar(im_id)
    plt.setp(ax6.get_yticklabels(), rotation=90, ha="center", va="baseline",
             rotation_mode="anchor")

    for i in range(2):
        for j in range(2):
            ax6.text(j, i, cf[i, j], ha="center", va="center", color="w")

    # Save the figure as an image.
    f.savefig(filename)
