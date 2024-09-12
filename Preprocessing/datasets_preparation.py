
"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This file contains classes and functions for network training.
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:

from config import Config
from loss_functions import DiceLoss
from os_tools import subdirs, sub_niftis

# System imports:

import numpy as np
import torch
from torch import nn
import torchio as tio
from torch.utils.data import DataLoader

import os
from os.path import join
from shutil import copyfile, move
from datetime import datetime
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import shutil




# Print configs:

np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)


# ----------------------------------------------- Yale Glioma Dataset ------------------------------------------------

# Set project root path:
# self.root = '/home/arman_avesta/nncapsnet'
path = os.getcwd()
if '/Users/arman/' in path:
    root = '/Users/arman/projects/sccapsnet'
elif '/Users/sa936/' in path:
    root = '/Users/sa936/projects/sccapsnet'
elif '/home/arman_avesta/' in path:
    root = '/home/arman_avesta/sccapsnet'
else:
    raise Exception('edit config.py --> set self.root')

data_folder = join(root, 'data')
images_folder = join(data_folder, 'images')



# Select image groups with more than 1 flair imagea nd 1 flair segmentation and copy them to image2 folder:

# destination =  join(data_folder, 'subjects_with_more_images')
#
# subjects = subdirs(images_folder, join=True)
# print(len(subjects))
#
# n_images = np.zeros(len(subjects))
# for i, subject in enumerate(tqdm(subjects)):
#     n_images[i] = len(sub_niftis(subject))
#
# print(np.unique(n_images))
#
# for subject in tqdm(subjects):
#     if len(sub_niftis(subject)) > 2:
#         shutil.move(subject, subject.replace('/images', '/images2'))





# Merge the labels of segmentations with more than 1 nifti file:

# To be done










#
#
#
#
#
# # rename files to img.nii.gz and seg.nii.gz:
#
#
# subjects_folders = subdirs(images_folder, join=True)
#
# for subject_folder in tqdm(subjects_folders):
#     nifti_files = sub_niftis(subject_folder, join=False)
#     name_lengths = np.array([len(nifti_file) for nifti_file in nifti_files])
#     img_index, seg_index = np.argmin(name_lengths), np.argmax(name_lengths)
#     shutil.move(join(subject_folder, nifti_files[img_index]), join(subject_folder, 'img.nii.gz'))
#     shutil.move(join(subject_folder, nifti_files[seg_index]), join(subject_folder, 'seg.nii.gz'))
#
#
#
# # check if imgs and segs have the same affine:
#
# destination = join(data_folder, 'images_with_inconsistent_attributes')
#
# subject_names = subdirs(images_folder, join=False)
# subjects = []
#
# subject_name = subject_names[0]
#
# for subject_name in tqdm(subject_names):
#     subject = tio.Subject(img=tio.ScalarImage(join(images_folder, subject_name, 'img.nii.gz')),
#                           seg=tio.LabelMap(join(images_folder, subject_name, 'seg.nii.gz')),
#                           name=subject_name)
#     try:
#         subject.check_consistent_affine()
#         subject.check_consistent_space()
#         subject.check_consistent_spatial_shape()
#         subject.check_consistent_orientation()
#     except RuntimeError as err:
#         print(subject_name)
#         shutil.move(join(images_folder, subject_name), join(destination, subject_name))
#


# split into training, validation, and test sets:

csvs_folder = join(data_folder, 'csvs')
split = {'train': .8, 'valid': .1, 'test': .1}


subjects = subdirs(images_folder, join=False)
np.random.shuffle(subjects)

n = len(subjects)
train_n = int(np.round(split['train'] * n))
valid_n = int(np.round(split['valid'] * n))
test_n = n - train_n - valid_n
print(f'''
total number of subjects:       {n}
number of training subjects:    {train_n}
number of validation subjects:  {valid_n}
number of test subjects:        {test_n}
''')

train_subjects = subjects[:train_n]
valid_subjects = subjects[train_n:train_n + valid_n]
test_subjects = subjects[train_n+valid_n:]

assert (len(train_subjects) == train_n) and (len(valid_subjects) == valid_n) and (len(test_subjects) == test_n)

train_df, valid_df, test_df = pd.DataFrame(train_subjects), pd.DataFrame(valid_subjects), pd.DataFrame(test_subjects)
os.makedirs(csvs_folder, exist_ok=True)
train_df.to_csv(join(csvs_folder, 'train_subjects.csv'), header=False, index=False)
valid_df.to_csv(join(csvs_folder, 'valid_subjects.csv'), header=False, index=False)
test_df.to_csv(join(csvs_folder, 'test_subjects.csv'), header=False, index=False)





# -------------------------------------------------- ADNI Dataset ----------------------------------------------------
#
# # Set project root path:
# # self.root = '/home/arman_avesta/nncapsnet'
# path = os.getcwd()
# if '/Users/arman/' in path:
#     root = '/Users/arman/projects/sccapsnet'
# elif '/Users/sa936/' in path:
#     root = '/Users/sa936/projects/sccapsnet'
# elif '/home/arman_avesta/' in path:
#     root = '/home/arman_avesta/sccapsnet'
# else:
#     raise Exception('edit config.py --> set self.root')
#
# data_folder = join(root, 'data')
# images_folder = join(data_folder, 'images')
# images2_folder = join(data_folder, 'images2')
# '''
# example images_folder:
# /Users/sa936/projects/sccapsnet/data/images
# '''
#
#
# # delete_phrase = 'aparc+aseg_brainbox2.nii.gz'
# # recursive_delete(images_folder, delete_phrase)
# # recursive_rename(images_folder, 'aparc+aseg_img.nii.gz', 'seg.nii.gz')
#
# '''
# example path for an img.nii.gz file:
# /Users/sa936/projects/sccapsnet/data/images/002_S_0295/2009-05-22_07_00_57.0/img.nii.gz
# '''
# subjects = subdirs(images_folder, join=False)
#
# subject = subjects[0]
# for subject in tqdm(subjects, desc='re-structuring folder tree'):        # example subject: 002_S_0295
#     scans = subdirs(join(images_folder, subject), join=False)
#     scan = scans[0]
#     for scan in scans:          # example scan: 2009-05-22_07_00_57.0
#         date = scan.split('_')[0]       # example date: 2009-05-22
#         origin_folder = join(images_folder, subject, scan)
#         destination_folder = join(images2_folder, f'{subject}__{date}')
#         shutil.move(origin_folder, destination_folder)








# LABEL REMAPPING:

# subjects = subdirs(images_folder)
#
# labels = []

# for i, subject in enumerate(tqdm(subjects)):
#     seg = tio.LabelMap(join(images_folder, subject, 'seg.nii.gz')).numpy()
#     unique_labels = np.unique(seg)
#     labels.append(unique_labels)
#
#
# ns = np.zeros(len(labels))
# for i, label in enumerate(labels):
#     ns[i] = len(labels[i])
#
# overall_labels = [   0,    2,    4,    5,    7,    8,   10,   11,   12,   13,   14,
#           15,   16,   17,   18,   24,   26,   28,   30,   31,   41,   43,
#           44,   46,   47,   49,   50,   51,   52,   53,   54,   58,   60,
#           62,   63,   72,   77,   80,   85,  251,  252,  253,  254,  255,
#         1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
#         1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021,
#         1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032,
#         1033, 1034, 1035, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
#         2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
#         2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029,
#         2030, 2031, 2032, 2033, 2034, 2035]
#
# for label in tqdm(labels):
#     for code in label:
#         if code not in overall_labels:
#             overall_labels.append(code)


# overall_labels2 = np.sort(overall_labels)

#
# '''
# 0: background
# 17 --> 1: left hippocampus
# 53 --> 2: right hippocampus
# 10 --> 3: left thalamus
# 49 --> 4: right thalamus
# 14 --> 5: 3rd ventricle
# '''
#
# labels_dict = {}
# for label in overall_labels2:
#     labels_dict[label] = 0
#
# labels_dict[17] = 1
# labels_dict[53] = 2
# labels_dict[10] = 3
# labels_dict[49] = 4
# labels_dict[14] = 5
#
# print(np.unique(list(labels_dict.values())))
#
# print(labels_dict)