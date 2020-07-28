# System imports.
import sys
import os
import datetime

# ML imports.
import numpy as np
import tensorflow as tf
from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt

# Local imports.
from loading import loadAugmentedBinaryDataset
from constants import GLOBAL_SEED
from createVGG16 import createVGG16

# Take note of the starting time of the program.
timestamp = datetime.datetime.now()

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = loadAugmentedBinaryDataset()

# Normalise the images of all three datasets.
X_train = tf.keras.utils.normalize(X_train)
X_val = tf.keras.utils.normalize(X_val)
X_test = tf.keras.utils.normalize(X_test)

# Set global seed for consistent run-to-run results.
tf.random.set_seed(GLOBAL_SEED)

# Create the model.
model = createVGG16(
    learning_rate=1e-5,
    l2_factor=0.01,
    dropout_factor=0.4
)

# Print the summary of the network.
print(model.summary())

# Setup checkpoint callback
checkpoint_path = "/cs/scratch/as521/models/checkpoints/basic-model-cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    period=25)

# Train the model, or load the weights.
if "--load" in sys.argv:
    # checkpoint_name = os.path.join(checkpoint_dir, sys.argv[sys.argv.index("--load") + 1])
    checkpoint_name = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(checkpoint_name)
else:
    # Train the model.
    history = model.fit(X_train, y_train, epochs=50, batch_size=3,
                        validation_data=(X_val, y_val), callbacks=[], verbose=2)

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
            text = ax6.text(j, i, cf[i, j], ha="center",
                            va="center", color="w")

    # Save the figure as an image.
    f.savefig('./results_vgg16_{}.png'.format(lr))

# Generate confusion matrix again, for evaluation and training dataset.
cf_text = ""
predictions = model.predict(X_val, batch_size=2)
predictions = [round(p[0]) for p in predictions]
cf = confusion_matrix(y_val, predictions)
cf_text += "Validation\n" + str(cf) + "\n"

predictions = model.predict(X_train, batch_size=2)
predictions = [round(p[0]) for p in predictions]
cf = confusion_matrix(y_train, predictions)
cf_text += "Training\n" + str(cf) + "\n"

# Save confusion matrices in text format into the file.
file = open('confusion_matrix_vgg16_{}.txt'.format(lr), 'w')
file.write(cf_text)
file.close()

print(cf_text)

# Print total execution time.
print("Total execution time: ", datetime.datetime.now() - timestamp)
