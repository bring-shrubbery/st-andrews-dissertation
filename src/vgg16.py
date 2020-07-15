import sys
import os
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.math import confusion_matrix
from tensorflow.keras.constraints import *
import matplotlib.pyplot as plt

import numpy as np

# Local imports
from loading import loadBinaryDataset

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = loadBinaryDataset()

X_train = tf.keras.utils.normalize(X_train)
X_val = tf.keras.utils.normalize(X_val)
X_test = tf.keras.utils.normalize(X_test)

regularization_factor = 0.3

# Create CNN structure
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(512, 512, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Dropout(regularization_factor))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Dropout(regularization_factor))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Dropout(regularization_factor))

model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Dropout(regularization_factor))

model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Dropout(regularization_factor))

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid',
                       bias_constraint=MinMaxNorm(min_value=0.1, max_value=0.9)))

print(model.summary())

# exit()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000001),
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(
              #     from_logits=True),
              # loss=tf.keras.losses.CategoricalCrossentropy(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', 'mse', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

print("Sizes:", X_train.shape, y_train.shape)
# exit()

# history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

# exit()

# Setup checkpoint callback

checkpoint_path = "/cs/scratch/as521/models/checkpoints/basic-model-cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    period=25)

# Train the model

history = None

if "--load" in sys.argv:
    # checkpoint_name = os.path.join(checkpoint_dir, sys.argv[sys.argv.index("--load") + 1])
    checkpoint_name = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(checkpoint_name)
else:
    history = model.fit(X_train, y_train, epochs=200, batch_size=4, validation_data=(
        X_val, y_val), callbacks=[cp_callback], verbose=1)

    f = plt.figure(figsize=(15, 10))
    ax1 = f.add_subplot(231, xlabel='Epoch', ylabel='Accuracy',  ylim=[0, 1])
    ax2 = f.add_subplot(232, xlabel='Epoch', ylabel='MSE',       ylim=[0, 1])
    ax3 = f.add_subplot(233, xlabel='Epoch', ylabel='Precision', ylim=[0, 1])
    ax4 = f.add_subplot(234, xlabel='Epoch', ylabel='Recall',    ylim=[0, 1])
    ax5 = f.add_subplot(235, xlabel='Epoch', ylabel='Loss',      ylim=[0, 1])

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

    f.savefig('./accuracy.png')

predictions = model.predict(X_test, batch_size=4)

predictions = [p[0] for p in predictions]

print(len(y_test))
print(len(predictions))
cf = confusion_matrix(y_test, predictions)

print(cf)

file = open('confusion_matrix.txt', 'w')
file.write(str(cf))
file.close()

# TODO:
# Plot confusion matrix
# Save models
# Write up