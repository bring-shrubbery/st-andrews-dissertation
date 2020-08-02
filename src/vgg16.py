print('Preparing...')

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
from tensorflow.data import Dataset

# Local imports
from loading import loadAugmentedBinaryDataset, loadAugmentedBinaryDatasetFromFiles
from createVGG16 import createVGG16
from plot_history import plotAndSave
from constants import GLOBAL_SEED, BASELINE_MODEL_PATH

from twitter_api import tweet

try:
    # Setup constants
    EPOCHS = 100
    BATCH_SIZE = 2
    LEARNING_RATE = 0.00001
    DROPOUT_FACTOR = 0.3
    L2_FACTOR = 0.1

    VERBOSITY_1 = 2
    VERBOSITY_2 = 1

    # Take note of the starting time of the program.
    timestamp = datetime.datetime.now()

    # Load dataset
    print('Loading dataset...')
    X_train, y_train, X_val, y_val, X_test, y_test = loadAugmentedBinaryDatasetFromFiles()

    # Set global seed for consistent run-to-run results.
    random.set_seed(GLOBAL_SEED)

    # Setup callbacks
    earlyStopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        verbose=VERBOSITY_2,
        mode='min'
    )
    mcp_save = ModelCheckpoint(
        BASELINE_MODEL_PATH,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        patience=10,
        verbose=VERBOSITY_2
    )
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=7,
        verbose=VERBOSITY_2,
        epsilon=1e-4,
        mode='min'
    )
    csv_logger = CSVLogger('./baseline_log.csv')

    # Create the model.
    print('Creating the model...')
    model = createVGG16(
        learning_rate=LEARNING_RATE,
        l2_factor=L2_FACTOR,
        dropout_factor=DROPOUT_FACTOR
    )

    # Print the summary of the network.
    print(model.summary())

    # Train the model.
    print('Training the model...')
    history = model.fit(
        X_train,
        y_train,
        # train_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        # validation_data=val_dataset,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss, csv_logger],
        verbose=VERBOSITY_1
    )

    plotAndSave('./baseline_results.png', history, model, X_val, y_val)

    print('Evaluating results...')
    # Generate confusion matrix again, for evaluation and training dataset.
    cf_text = ""
    predictions = model.predict(X_val, batch_size=BATCH_SIZE)
    predictions = [round(p[0]) for p in predictions]
    cf = confusion_matrix(y_val, predictions)
    cf_text += "Validation\n" + str(cf.numpy())[1:-1].replace('\n ', '\n') + "\n"

    predictions = model.predict(X_train, batch_size=BATCH_SIZE)
    predictions = [round(p[0]) for p in predictions]
    cf = confusion_matrix(y_train, predictions)
    cf_text += "Training\n" + str(cf.numpy())[1:-1].replace('\n ', '\n') + "\n"

    # Save confusion matrices in text format into the file.
    file = open('vgg16_confusion_matrix.txt', 'w')
    file.write(cf_text)
    file.close()

    print(cf_text)

    # Print total execution time.
    dat_dur = datetime.datetime.now() - timestamp
    print("Total execution time: ", dat_dur)

    print('Final evalutaiton...')
    evaluation = model.evaluate(
        X_val, y_val, batch_size=BATCH_SIZE, verbose=VERBOSITY_1, return_dict=True)
    print(evaluation)
    val_acc = int(evaluation['accuracy']*100)

    duration = dat_dur.seconds // 60  # Minutes (rounded)

    if duration > 60:
        duration = str(int(duration / 6) / 10) + ' hrs'
    else:
        duration = str(int(duration * 10) / 10) + ' mins'

    tweet("@bring_shrubbery \nDone.\nTime: {}\nAcc: .{}\n{}".format(
        duration,
        val_acc,
        cf_text
    ), "baseline_results.png")
except:
    tweet("@bring_shrubbery Latest execution crashsed at {}".format(datetime.datetime.now()))
