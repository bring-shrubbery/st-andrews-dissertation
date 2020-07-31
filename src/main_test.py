print('Preparing...')
from tensorflow.keras.models import load_model
from constants import BASELINE_MODEL_PATH, GLOBAL_SEED
from loading import loadAugmentedBinaryDatasetFromFiles

print('Loading dataset...')
_, _, _, _, X_test, y_test = loadAugmentedBinaryDatasetFromFiles()

print('Loading model...')
model = load_model(BASELINE_MODEL_PATH)

print('Evaluating model on testing dataset...')
results = model.evaluate(X_test, y_test, batch_size=4, return_dict=True)

print('Results:')
print(results)
