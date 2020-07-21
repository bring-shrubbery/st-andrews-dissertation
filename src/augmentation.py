import numpy as np

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

  for i in range(image_set_side):
    noise = np.random.normal(0, 500, image_set[i].shape)
    noisy_image = image_set[i] + noise
    new_X.append(noisy_image)
    new_y.append(image_set_y[i])

  new_X = np.array(new_X)
  new_y = np.array(new_y)

  return new_X, new_y
