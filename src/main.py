# System imports
import sys
import os
import datetime

# ML imports
import numpy as np
from tensorflow.math import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow import random

# Local imports
from loading import loadAugmentedBinaryDataset
from constants import GLOBAL_SEED
from createBaseline import createBaseline
from plot_history import plotAndSave

from twitter_api import tweet

# Setup constants
EPOCHS = 1
BATCH_SIZE = 2
LEARNING_RATE = 0.00002
DROPOUT_FACTOR = 0.45
L2_FACTOR = 0.15

# Take note of the starting time of the program
timestamp = datetime.datetime.now()

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = loadAugmentedBinaryDataset(normalize=True)

# Set global seed for consistent run-to-run results
random.set_seed(GLOBAL_SEED)

# Setup callbacks
checkpoint_path = "/cs/scratch/as521/models/checkpoints/baseline-model.h5"

earlyStopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min')
mcp_save = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', patience=7, verbose=1)
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
csv_logger = CSVLogger('./baseline-log.csv')

# Create global model.
model = None

# Train the model first if there's no '--load' flag.
if "--load" in sys.argv:
    # Load the model
    model = load_model(checkpoint_path)
else:
    # Create the model
    model = createBaseline(learning_rate=LEARNING_RATE, dropout_factor=DROPOUT_FACTOR, l2_factor=L2_FACTOR)

    # Print the summary of the network
    print(model.summary())

    # Train the model.
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss, csv_logger],
        verbose=2
    )

    # Load the best model saved by the ModelCheckpoint callback
    print("Loading best model...")
    model = load_model(checkpoint_path)

    plotAndSave('./results.png', history, model, X_val, y_val)

# Generate confusion matrix again, for evaluation and training dataset.
cf_text = ""
predictions = model.predict(X_val, batch_size=4)
predictions = [round(p[0]) for p in predictions]
cf = confusion_matrix(y_val, predictions)
cf_text += "Validation:\n" + str(cf.numpy())[1:-1].replace('\n ', '\n') + "\n"

predictions = model.predict(X_train, batch_size=4)
predictions = [round(p[0]) for p in predictions]
cf = confusion_matrix(y_train, predictions)
cf_text += "Training:\n" + str(cf.numpy())[1:-1].replace('\n ', '\n') + "\n"

# Save confusion matrices in text format into the file.
file = open('confusion_matrix.txt', 'w')
file.write(cf_text)
file.close()

print(cf_text)

# Print total execution time.
print("Total execution time: ", datetime.datetime.now() - timestamp)

evaluation = model.evaluate(X_val, y_val, batch_size=2, verbose=2, return_dict=True)
print(evaluation)
val_acc = evaluation['accuracy']

tweet("@bring_shrubbery Done. Time: {}.\nValAcc: {}\n{}".format(
    datetime.datetime.now() - timestamp,
    val_acc,
    cf_text
))
