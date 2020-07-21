import sys
import os
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.math import confusion_matrix
from tensorflow.keras.constraints import *
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

# import tensorflow_addons as tfa

import numpy as np

# Local imports
from loading import loadAugmentedBinaryDataset

import datetime

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = loadAugmentedBinaryDataset()

# Trip to the size of the confirmed polyp images.
# no_polyp_pixel_data = no_polyp_pixel_data[:total_count]

# exit()

timestamp = datetime.datetime.now()

X_train = tf.keras.utils.normalize(X_train)
X_val = tf.keras.utils.normalize(X_val)
X_test = tf.keras.utils.normalize(X_test)

dropout_factor = 0.4

l2_factor = 0.1

# Create CNN
model = models.Sequential()
model.add(layers.Conv2D(96, (11, 11), input_shape=(512, 512, 1),
                        kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))

model.add(layers.BatchNormalization())

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D((3, 3)))






model.add(layers.Conv2D(384, (5, 5),
                        kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))

model.add(layers.Activation('relu'))


model.add(layers.MaxPooling2D((3, 3)))






model.add(layers.Conv2D(384, (3, 3), activation='relu',
                        kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))

model.add(layers.Conv2D(256, (3, 3), activation='relu',
                        kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))

model.add(layers.Conv2D(256, (3, 3),
                        kernel_regularizer=l2(l2_factor), bias_regularizer=l2(l2_factor)))
model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D((3, 3)))



model.add(layers.Dropout(dropout_factor))



model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(dropout_factor))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(dropout_factor))
model.add(layers.Dense(1, activation='sigmoid',
                       bias_constraint=MinMaxNorm(min_value=0.1, max_value=0.9)))

print(model.summary())

# exit()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
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
    period=100)

# Train the model

history = None

if "--load" in sys.argv:
    # checkpoint_name = os.path.join(checkpoint_dir, sys.argv[sys.argv.index("--load") + 1])
    checkpoint_name = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(checkpoint_name)
else:
    try:
        history = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(
        X_val, y_val), callbacks=[cp_callback], verbose=1)

    except:
        print("Error has occured while training")

    f = plt.figure(figsize=(15, 10))
    f.tight_layout()

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

    predictions = model.predict(X_val, batch_size=2)
    predictions = [round(p[0]) for p in predictions]
    print("Predictions:", predictions)
    print("Y_val:", y_val)
    cf = confusion_matrix(y_val, predictions)
    cf = np.array(cf)

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

    f.savefig('./accuracy.png')

predictions = model.predict(X_val, batch_size=4)
predictions = [round(p[0]) for p in predictions]
cf = confusion_matrix(y_val, predictions)

print(cf)

file = open('confusion_matrix.txt', 'w')
file.write("Validation\n"+str(cf))
file.close()

predictions = model.predict(X_train, batch_size=4)
predictions = [round(p[0]) for p in predictions]
cf = confusion_matrix(y_train, predictions)

print(cf)

# Only use validation  set for testing
# Get confusion matrix on training set.
# Add regularization into loss  function
# Add results into report
# Explain all the steps and reasoning done during the implementation

# TODO:
# Plot confusion matrix
# Save models
# Use VGG16 architecture
# Implement several general purpose classifier using transfer learning.
# Write up

print("Total execution time: ", datetime.datetime.now() - timestamp)
