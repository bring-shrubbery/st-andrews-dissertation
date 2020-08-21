# St Andrews MSc Artificial Intelligence Dissertation

This is the code to go with my dissertation at the University of St Andrews.

Due to not having git-lfs storage the `generated-features` folder and the model are not included. These can be genrated again from the full dataset using the provided scripts or downloaded from the links below.

The generated features folder is zipped and uploaded to Google Drive for that reason and uploaded to this [link](https://drive.google.com/file/d/1EL8BSWqIdEAcSaIS6aXdHkAw1u8tbCnZ/view?usp=sharing). The folders from inside of the downloaded zip file should be put into the `generated-features` directory in the project

The model is also not uploaded due to size restrictions, it was uploaded to Google Drive and is available through this [link](https://drive.google.com/file/d/19u8PWfp-Mywomg_gZmSgc7kzaSt8xhnW/view?usp=sharing). The downloaded `.h5` model should be put into the `models` directory.

## Description

Since the full dataset is too big to provide with the code, there is a smaller version, with just the right images extracted into separate filles. These images are stored in `generated-features` folder. The images are stored in numpy file format. There are 3 separate folders for training, validation and testing datasets and each folder contains another 2 folders for positive and negative samples. These images are loaded automatically by the `loadAugmentedBinaryDatasetFromFiles` with an option to normalise the loaded images by providing 'normalize=True' parameter.

## Running the code

All the code should be executed from inside `src` folder.

There are multiple things you can do with the provided code. You can train the baseline network by running:

```python3 main.py```

You can train the custom VGG16 model by running:

```python3 vgg16.py```

You can run one of the transfer learning models by running:

```python3 transfer_models.py```

You can evaluate the model saved in the `models` folder by running following command and providing name of the model as the parameter, for example, to run the best model that was achieved, run:

```python3 main_evaluate.py baseline-model.h5```

## File structure

In `features` folder: Annotations in CSV format.

In `generated-features` folder: Extracted smaller datasets from the full one.

In `models` folder: contains best models that were trained.

In results folder:
- `baseline` - contains the graphs, logs, etc. for the baseline model
- `filters` - contains filters extracted by `visualise_filters.py` script
- `vgg16` - contains graphs, logs, etc. for the custom VGG16 model

In `src` folder:
- `augmentation.py` - Contains functions that augment data samples as described in the report
- `constants.py` - Contains constants that are imported throughout the project
- `createBaseline.py` - Contains function which creates the baseline Keras model
- `createVGG16.py` - Contains function which create the custom VGG16 model
- `extract_images.ipynb` - Is a jupyter notebook which generates the files in `generated-features` if the full dataset is available
- `loading.py` - Contains all the loading functions
- `main.py` - Contains training script for the baseline model
- `main_evaluate` - Evaluates provided model on all 3 sets
- `plot_history` - Plots the graph that can be seen in the results fodler
- `transfer_models.py` - Will train the transfer learning models
- `vgg16.py` -  Will train the custom VGG16 model
- `visualise_filters.py` - Will visualise the first layer filters of the provided model and save them into `./results/filters` directory
