print('Preparing...')
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
from constants import MODEL_CHECKPOINT_DIR, GLOBAL_SEED
from loading import loadAugmentedBinaryDatasetFromFiles

# Check if file exists
full_path = MODEL_CHECKPOINT_DIR + sys.argv[-1]
if not os.path.isfile(full_path):
    print("No file {}".format(full_path))
    exit()

print('Loading dataset...')
X_train, y_train, X_val, y_val, X_test, y_test = loadAugmentedBinaryDatasetFromFiles()

print('Loading model...')
model = load_model(full_path)

print('Evaluating model on validation dataset...')
results = model.evaluate(X_val, y_val, batch_size=4, return_dict=True, verbose=0)

# Define print results function
def printResults(res):
    for metric in res:
        print("  {}: {}".format(metric, res[metric]))

print('Validation Results:')
printResults(results)

print('Evaluating model on training dataset...')
results = model.evaluate(X_train, y_train, batch_size=4, return_dict=True, verbose=0)

print('Training Results:')
printResults(results)

print('Evaluating model on testing dataset...')
results = model.evaluate(X_test, y_test, batch_size=4, return_dict=True, verbose=0)

print('Testing Results:')
printResults(results)
