import tensorflow as tf
from typing import List
import pydicom
import pandas as pd
import numpy as np
from os.path import join
import os
import re
from constants import DATASET_ROOT_DIRECTORY
from random import randrange
from augmentation import flipImagesLR, flipImagesUD, addGaussianNoise, augmentTranslation, augmentRotation
from constants import GLOBAL_SEED

# MARK: Dicom file loaders


def loadDicom(fullpath: str):
    return pydicom.dcmread(fullpath)


def loadDicomList(filepathList: List[str]):
    return [loadDicom(p) for p in filepathList]


def loadDicomPixelData(fullpath: str):
    return loadDicom(fullpath).pixel_array


def loadDicomListPixelData(filepathList: List[str]):
    return [f.pixel_array for f in loadDicomList(filepathList)]

# MARK: Feature loaders


def loadLargePolypFeatures(filepath):
    large_polyps_df = pd.read_csv(filepath)

    # Clean main column names
    large_polyps_df = large_polyps_df.rename(columns={
        'TCIA Number': 'id',
        'Slice# polyp Supine': 'supine_slice',
        'Slice# polyp Prone': 'prone_slice'
    })

    # Clean up supine and prone slice values.
    large_polyps_df.supine_slice = [s[10:].strip()
                                    for s in large_polyps_df.supine_slice]
    large_polyps_df.supine_slice = ["nan" if len(
        s) == 0 else s for s in large_polyps_df.supine_slice]

    large_polyps_df.prone_slice = [str(s) for s in large_polyps_df.prone_slice]

    # Filter dataframe to exclude rows that do not contian at least one slice number
    prone_qualifier = large_polyps_df.prone_slice.str.contains('nan')
    supine_qualifier = large_polyps_df.supine_slice.str.contains('nan')
    large_polyps_df = large_polyps_df[~(prone_qualifier & supine_qualifier)]

    return large_polyps_df


def loadMediumPolypFeatures(filepath):
    medium_polyps_df = pd.read_csv(filepath)

    # Clean main column names
    medium_polyps_df = medium_polyps_df.rename(columns={
        'TCIA Number': 'id',
        'Slice# polyp Supine': 'supine_slice',
        'Slice# polyp Prone': 'prone_slice'
    })

    # Clean up supine and prone slice values.
    # medium_polyps_df.supine_slice = ["nan" if len(s) == 0 else s for s in medium_polyps_df.supine_slice]

    medium_polyps_df.supine_slice = [str(s) for s in medium_polyps_df.supine_slice]
    medium_polyps_df.prone_slice = [str(s) for s in medium_polyps_df.prone_slice]

    # Filter dataframe to exclude rows that do not contian at least one slice number
    prone_qualifier = medium_polyps_df.prone_slice.str.contains('nan')
    supine_qualifier = medium_polyps_df.supine_slice.str.contains('nan')
    medium_polyps_df = medium_polyps_df[~(prone_qualifier & supine_qualifier)]

    return medium_polyps_df


def loadLargePolypsImageIds(filepath):
    df = loadLargePolypFeatures(filepath)
    return filterSliceIds(df)


def loadMediumPolypsImageIds(filepath):
    df = loadMediumPolypFeatures(filepath)
    return filterSliceIds(df)


def filterSliceIds(dataframe):
    files = []

    for i in dataframe.index:
        row = dataframe.loc[i]

        tcia_id = row.id
        supine_slice = row.supine_slice
        prone_slice = row.prone_slice

        for s in supine_slice.split('/'):
            if s != 'nan':
                files.append({
                    'tcia_id': tcia_id,
                    'supine_id': s.strip()
                })

        for s in prone_slice.split('/'):
            if s != 'nan':
                files.append({
                    'tcia_id': tcia_id,
                    'prone_id': s.strip()
                })

    return files


def formatDicomFilename(id):
    return "1-"+str(id).rjust(3, '0')+".dcm"


def loadLargePolypData():
    image_data_list = []

    large_polyp_csv_file = join(os.path.dirname(
        __file__), '../features/large_polyps.csv')
    # print(large_polyp_csv_file)
    files = loadLargePolypsImageIds(large_polyp_csv_file)

    # Loop over the processed rows of spreadsheet.
    for f in files:
        full_path = join(DATASET_ROOT_DIRECTORY, 'large-polyps', f['tcia_id'])
        dirlist = os.listdir(full_path)

        # Skip directories with two studies.
        if len(dirlist) > 1:
            continue

        next_dir = join(full_path, dirlist[0])

        dirlist = os.listdir(next_dir)

        for subpath in dirlist:
            p = join(next_dir, subpath)
            subfiles = os.listdir(p)

            # Skip folders with one image.
            if len(subfiles) < 10:
                continue

            # Identify supine folders
            if re.search('supine', subpath, re.IGNORECASE) and 'supine_id' in f:
                # print('Supine:', f['supine_id'])

                filename = formatDicomFilename(f['supine_id'])
                dicom_full_path = join(p, filename)
                # print(dicom_full_path)

                image_data_list.append({
                    'study_id': f['tcia_id'],
                    'slice_id': f['supine_id'],
                    'path': dicom_full_path
                })

            # Identify supine folders
            if re.search('prone', subpath, re.IGNORECASE) and 'prone_id' in f:
                # print('Prone:', f['prone_id'])

                filename = formatDicomFilename(f['prone_id'])
                dicom_full_path = join(p, filename)
                # print(dicom_full_path)

                image_data_list.append({
                    'study_id': f['tcia_id'],
                    'slice_id': f['prone_id'],
                    'path': dicom_full_path
                })

    return image_data_list


def loadMediumPolypData():
    image_data_list = []

    medium_polyp_csv_file = join(os.path.dirname(
        __file__), '../features/medium_polyps.csv')
    # print(large_polyp_csv_file)
    files = loadMediumPolypsImageIds(medium_polyp_csv_file)

    # Loop over the processed rows of spreadsheet.
    for f in files:
        full_path = join(DATASET_ROOT_DIRECTORY, 'medium-polyps', f['tcia_id'])
        dirlist = os.listdir(full_path)

        # Skip directories with two studies.
        if len(dirlist) > 1:
            continue

        next_dir = join(full_path, dirlist[0])

        dirlist = os.listdir(next_dir)

        for subpath in dirlist:
            p = join(next_dir, subpath)
            subfiles = os.listdir(p)

            # Skip folders with one image.
            if len(subfiles) < 10:
                continue

            # Identify supine folders
            if re.search('supine', subpath, re.IGNORECASE) and 'supine_id' in f:
                # print('Supine:', f['supine_id'])

                filename = formatDicomFilename(f['supine_id'])
                dicom_full_path = join(p, filename)
                # print(dicom_full_path)

                image_data_list.append({
                    'study_id': f['tcia_id'],
                    'slice_id': f['supine_id'],
                    'position': 'supine',
                    'combined_name': '{}-{}-{}'.format(f['tcia_id'], f['supine_id'], 'supine'),
                    'path': dicom_full_path
                })

            # Identify supine folders
            if re.search('prone', subpath, re.IGNORECASE) and 'prone_id' in f:
                # print('Prone:', f['prone_id'])

                filename = formatDicomFilename(f['prone_id'])
                dicom_full_path = join(p, filename)
                # print(dicom_full_path)

                image_data_list.append({
                    'study_id': f['tcia_id'],
                    'slice_id': f['prone_id'],
                    'position': 'prone',
                    'combined_name': '{}-{}-{}'.format(f['tcia_id'], f['prone_id'], 'prone'),
                    'path': dicom_full_path
                })

    return image_data_list


def loadBinaryDataset():
    large_polyp_data_list = loadLargePolypData()
    medium_polyp_data_list = loadMediumPolypData()

    image_data_list = large_polyp_data_list + medium_polyp_data_list

    datasets = [
        [],  # X_train
        [],  # y_train
        [],  # X_val
        [],  # y_val
        [],  # X_test
        []  # y_test
    ]

    # Reindex the links by the study id.
    study_based_map = {}
    for i in range(len(image_data_list)):
        sample = image_data_list[i]
        sid = sample['study_id']

        if sid in study_based_map:
            study_based_map[sid].append(sample['path'])
        else:
            study_based_map[sid] = [sample['path']]

    val_fraction = 1./.2
    test_fraction = 1./.2

    # Only append the images from the same study to the same dataset.
    # Samples are added to the dataset with the least weighted sample sount.
    for key in study_based_map:
        min_size = int(min([
            len(datasets[0]),
            len(datasets[1]),
            len(datasets[2])*val_fraction,
            len(datasets[3])*val_fraction,
            len(datasets[4])*test_fraction,
            len(datasets[5])*test_fraction
        ]))
        for i in range(0, len(datasets), 2):
            dataset_weight = val_fraction if i == 2 else (
                test_fraction if i == 4 else 1.)
            if int(len(datasets[i])*dataset_weight) == min_size:
                datasets[i].extend(study_based_map[key])
                datasets[i+1].extend([1]*len(study_based_map[key]))
                break

    X_train = loadDicomListPixelData(datasets[0])
    y_train = datasets[1]

    X_val = loadDicomListPixelData(datasets[2])
    y_val = datasets[3]

    X_test = loadDicomListPixelData(datasets[4])
    y_test = datasets[5]

    # Reshape images to be 512x512
    X_train = [x.reshape(512, 512, 1) for x in X_train]
    X_val = [x.reshape(512, 512, 1) for x in X_val]
    X_test = [x.reshape(512, 512, 1) for x in X_test]

    # Append no-polyp samples
    # all_no_polyp_images = loadRandomNoPolypImages()
    slice_ids = sorted([p['slice_id'] for p in image_data_list])
    all_no_polyp_images = loadNoPolypImagesFromSlideIds(slice_ids)
    all_no_polyp_images = [x.reshape(512, 512, 1) for x in all_no_polyp_images]

    total_count = len(X_train)+len(X_val)+len(X_test)
    X_train_len = len(X_train)
    X_train_val_len = len(X_train) + len(X_val)

    # Decide which set has least entries and add new subset there.
    for i in range(total_count):
        if i < X_train_len:
            X_train.append(all_no_polyp_images[i])
            y_train.append(0)
        elif i < X_train_val_len:
            X_val.append(all_no_polyp_images[i])
            y_val.append(0)
        else:
            X_test.append(all_no_polyp_images[i])
            y_test.append(0)

    # Print datset sizes.
    print(len(X_train))
    print(len(y_train))
    print(len(X_val))
    print(len(y_val))
    print(len(X_test))
    print(len(y_test))

    # Convert all data lists into numpy arrays for output.
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def loadRandomNoPolypImages():
    image_paths = []

    no_polyp_csv_file = join(os.path.dirname(
        __file__), '../features/no_polyps.csv')

    no_polyp_df = pd.read_csv(no_polyp_csv_file).rename(columns={
        'TCIA Patient ID': 'id'
    })

    # print(no_polyp_df)

    images_to_sample_from_single_patient = 1

    for i in no_polyp_df.index:
        tcia_id = no_polyp_df.loc[i].id

        full_path = join(DATASET_ROOT_DIRECTORY, 'no-polyps', tcia_id)
        if not os.path.exists(full_path):
            continue
        dirlist = os.listdir(full_path)

        if len(dirlist) > 1:
            continue

        full_path = join(full_path, dirlist[0])
        dirlist = os.listdir(full_path)

        for subpath in dirlist:
            p = join(full_path, subpath)
            files = os.listdir(p)
            if len(files) > 10:

                for i in range(images_to_sample_from_single_patient):
                    # Get random image with index betweeon 100 and 400.
                    rand_ind = int(randrange(50, min(350, len(files)-1), 1))
                    image_paths.append(join(p, files[rand_ind]))

                break

    return loadDicomListPixelData(image_paths)


def getNoPolypDataframe():
    csv_filename = os.path.join(os.path.dirname(
        __file__), '../features/no_polyps.csv')
    name_map = {'TCIA Patient ID': 'id'}
    return pd.read_csv(csv_filename).rename(columns=name_map)


def loadNoPolypImagesFromSlideIds(ids):
    ids = sorted(int(s) for s in ids)

    no_polyp_df = getNoPolypDataframe()

    image_paths = []

    for i in no_polyp_df.index:
        # Get study id and create a path to it in the dataset
        tcia_id = no_polyp_df.loc[i].id
        full_path = os.path.join(DATASET_ROOT_DIRECTORY, 'no-polyps', tcia_id)

        # Escape if the folder does not exist (because
        # we might have not downloaded full dataset)
        if not os.path.exists(full_path):
            continue

        # Get list of subfolders
        dirlist = os.listdir(full_path)

        # Skip folder if there are multiple studies, for simplicity
        if len(dirlist) > 1:
            continue

        # Navigate into the first folder and list the items inside
        full_path = os.path.join(full_path, dirlist[0])
        dirlist = os.listdir(full_path)

        # Iterate over the subpaths (which contain supine and prone positions)
        for subpath in dirlist:
            # Get the path to the particular prone/supine path
            p = os.path.join(full_path, subpath)

            # Get filenames inside and only proceed if there are more than
            # 10 files to eliminate folders with one sample
            files = os.listdir(p)
            if len(files) > 10:
                files_extracted = 0
                files_to_extract = 2
                _id = -1
                while files_extracted < files_to_extract:
                    if len(ids) == 0:
                        break

                    # Proceed if the filename is in the file list.
                    current_name = formatDicomFilename(str(ids[_id]))
                    if current_name in files:
                        # File exists!
                        full_file_path = os.path.join(p, current_name)
                        image_paths.append(full_file_path)
                        ids.pop(_id)
                        files_extracted += 1
                    else:
                        _id -= 1
            # Only use one folder per patient
            # break
            if len(ids) == 0:
                break

        if len(ids) == 0:
            break

    return loadDicomListPixelData(image_paths)


def loadAugmentedBinaryDataset():
    X_train, y_train, X_val, y_val, X_test, y_test = loadBinaryDataset()

    # Generate flipped images
    flipped_lr_X_train, flipped_lr_y_train = flipImagesLR(X_train, y_train)
    flipped_ud_X_train, flipped_ud_y_train = flipImagesUD(X_train, y_train)

    flipped_lr_X_val, flipped_lr_y_val = flipImagesLR(X_val, y_val)
    flipped_ud_X_val, flipped_ud_y_val = flipImagesUD(X_val, y_val)

    flipped_lr_X_test, flipped_lr_y_test = flipImagesLR(X_test, y_test)
    flipped_ud_X_test, flipped_ud_y_test = flipImagesUD(X_test, y_test)

    # Concatenate all generated features into one array.
    new_X_train = list(flipped_lr_X_train) + list(flipped_ud_X_train)
    new_y_train = list(flipped_lr_y_train) + list(flipped_ud_y_train)

    new_X_val = list(flipped_lr_X_val) + list(flipped_ud_X_val)
    new_y_val = list(flipped_lr_y_val) + list(flipped_ud_y_val)

    new_X_test = list(flipped_lr_X_test) + list(flipped_ud_X_test)
    new_y_test = list(flipped_lr_y_test) + list(flipped_ud_y_test)

    # Generate rotated images
    rotated_X_train, rotated_y_train = augmentRotation(X_train, y_train)
    rotated_X_val, rotated_y_val = augmentRotation(X_val, y_val)
    rotated_X_test, rotated_y_test = augmentRotation(X_test, y_test)

    # Generate translated images
    translated_X_train, translated_y_train = augmentTranslation(X_train, y_train)
    translated_X_val, translated_y_val = augmentTranslation(X_val, y_val)
    translated_X_test, translated_y_test = augmentTranslation(X_test, y_test)

    # Add rotated features
    new_X_train = new_X_train + list(rotated_X_train)
    new_y_train = new_y_train + list(rotated_y_train)

    new_X_val = new_X_val + list(rotated_X_val)
    new_y_val = new_y_val + list(rotated_y_val)

    new_X_test = new_X_test + list(rotated_X_test)
    new_y_test = new_y_test + list(rotated_y_test)

    # Add translated features
    new_X_train = new_X_train + list(translated_X_train)
    new_y_train = new_y_train + list(translated_y_train)

    new_X_val = new_X_val + list(translated_X_val)
    new_y_val = new_y_val + list(translated_y_val)

    new_X_test = new_X_test + list(translated_X_test)
    new_y_test = new_y_test + list(translated_y_test)

    # Add gaussian noise to all the features.
    noise_X_train, noise_y_train = addGaussianNoise(new_X_train, new_y_train)
    noise_X_val, noise_y_val = addGaussianNoise(new_X_val, new_y_val)
    noise_X_test, noise_y_test = addGaussianNoise(new_X_test, new_y_test)

    # Add noise features to the existing features.
    new_X_train = new_X_train + list(noise_X_train)
    new_y_train = new_y_train + list(noise_y_train)

    new_X_val = new_X_val + list(noise_X_val)
    new_y_val = new_y_val + list(noise_y_val)

    new_X_test = new_X_test + list(noise_X_test)
    new_y_test = new_y_test + list(noise_y_test)

    # Print sizes of the new sets
    print(len(new_X_train))
    print(len(new_y_train))
    print(len(new_X_val))
    print(len(new_y_val))
    print(len(new_X_test))
    print(len(new_y_test))

    # Convert back to np.array for output
    new_X_train = np.array(new_X_train)
    new_y_train = np.array(new_y_train)
    new_X_val = np.array(new_X_val)
    new_y_val = np.array(new_y_val)
    new_X_test = np.array(new_X_test)
    new_y_test = np.array(new_y_test)

    return new_X_train, new_y_train, new_X_val, new_y_val, new_X_test, new_y_test


