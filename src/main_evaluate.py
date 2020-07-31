print('Preparing...')
from tensorflow.keras.models import load_model
from constants import BASELINE_MODEL_PATH, GLOBAL_SEED
from loading import loadAugmentedBinaryDatasetFromFiles

print('Loading dataset...')
_, _, X_val, y_val, _, _ = loadAugmentedBinaryDatasetFromFiles()

print('Loading model...')
model = load_model(BASELINE_MODEL_PATH)

print('Evaluating model on validation dataset...')
results = model.evaluate(X_val, y_val, batch_size=4, return_dict=True)

print('Results:')
print(results)
