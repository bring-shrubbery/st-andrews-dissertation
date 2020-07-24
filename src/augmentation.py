import numpy as np
from constants import GLOBAL_SEED

def flipImagesLR(image_set, image_set_y):
  image_set_side = len(image_set)
  new_X = []
  new_y = []

  for i in range(image_set_side):
    flipped_lr_image = np.fliplr(image_set[i])
    new_X.append(flipped_lr_image)
    new_y.append(image_set_y[i])

  new_X = np.array(new_X)
  new_y = np.array(new_y)

  return new_X, new_y


def flipImagesUD(image_set, image_set_y):
  image_set_side = len(image_set)
  new_X = []
  new_y = []

  for i in range(image_set_side):
    flipped_lr_image = np.flipud(image_set[i])
    new_X.append(flipped_lr_image)
    new_y.append(image_set_y[i])

  new_X = np.array(new_X)
  new_y = np.array(new_y)

  return new_X, new_y

def addGaussianNoise(image_set, image_set_y):
  image_set_side = len(image_set)
  new_X = []
  new_y = []

  np.random.seed(GLOBAL_SEED)
  for i in range(image_set_side):
    noise = np.random.normal(0, 250, image_set[i].shape)
    noisy_image = image_set[i] + noise
    new_X.append(noisy_image)
    new_y.append(image_set_y[i])

  new_X = np.array(new_X)
  new_y = np.array(new_y)

  return new_X, new_y

def augmentRotation(image_set, image_set_y):
  image_set_side = len(image_set)
  new_X = []
  new_y = []

  for i in range(image_set_side):
    rotated_ccw = np.rot90(image_set[i])
    rotated_cw = np.rot90(image_set[i], 3)
    rotated_180 = np.rot90(image_set[i], 2)
    new_X.extend([rotated_ccw, rotated_cw, rotated_180])
    new_y.extend([image_set_y[i], image_set_y[i], image_set_y[i]])

  new_X = np.array(new_X)
  new_y = np.array(new_y)

  return new_X, new_y


def augmentTranslation(image_set, image_set_y, offset_x=0.25, offset_y=0.25):
  image_set_side = len(image_set)
  new_X = []
  new_y = []

  for i in range(image_set_side):
    translated_x_pos = np.roll(image_set[i], int(image_set[i].shape[0]*offset_x), axis=1)
    translated_y_pos = np.roll(image_set[i], int(image_set[i].shape[1]*offset_y), axis=0)
    translated_x_neg = np.roll(image_set[i], -int(image_set[i].shape[0]*offset_x), axis=1)
    translated_y_neg = np.roll(image_set[i], -int(image_set[i].shape[1] * offset_y), axis=0)
    
    new_X.extend([translated_x_pos, translated_y_pos, translated_x_neg, translated_y_neg])
    new_y.extend([image_set_y[i], image_set_y[i], image_set_y[i], image_set_y[i]])

  new_X = np.array(new_X)
  new_y = np.array(new_y)

  return new_X, new_y
