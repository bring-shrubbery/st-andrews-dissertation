from typing import List
import pydicom
import pandas as pd
from os.path import join
import os
import re
from constants import DATASET_ROOT_DIRECTORY

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


def loadLargePolypsImageIds(filepath):
    df = loadLargePolypFeatures(filepath)
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

def loadLargePolypImages():
    image_paths = []

    large_polyp_csv_file = join(os.path.dirname(__file__), '../features/large_polyps.csv')
    # print(large_polyp_csv_file)
    files = loadLargePolypsImageIds(large_polyp_csv_file)

    # Loop over the processed rows of spreadsheet.
    for f in files:
        full_path = join(DATASET_ROOT_DIRECTORY, 'large-polyps', f['tcia_id'])
        dirlist = os.listdir(full_path)
        
        # Skip directories with two studies.
        if len(dirlist) > 1: continue

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

                image_paths.append(dicom_full_path)

            # Identify supine folders
            if re.search('prone', subpath, re.IGNORECASE) and 'prone_id' in f:
                # print('Prone:', f['prone_id'])

                filename = formatDicomFilename(f['prone_id'])
                dicom_full_path = join(p, filename)
                # print(dicom_full_path)

                image_paths.append(dicom_full_path)

    return loadDicomList(image_paths)

            
