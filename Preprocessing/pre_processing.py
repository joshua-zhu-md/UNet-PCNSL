
"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This file contains pre-processing functions and classes.
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:
from config import Config
from os_tools import convert_csv_to_list, subdirs, print_and_log, log, recursive_delete
from image_tools import imgshow

# System imports:
import os
from os.path import join, exists
import shutil
import numpy as np
import torch
import torchio as tio
import nibabel as nib
import SimpleITK as sitk
from datetime import datetime
from tqdm import tqdm, trange
from multiprocessing import cpu_count, Pool

# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)

# ------------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------

def calculate_median_voxsize(subjects):
    voxsizes = np.zeros((len(subjects), 3))
    for i, subject in enumerate(tqdm(subjects, desc='calculating median voxel size')):
        voxsizes[i, :] = subject.img.spacing
    median_voxsize = tuple(np.median(voxsizes, axis=0))
    print(f'Median voxel size: {median_voxsize}')
    return median_voxsize


def calculate_max_shape(subjects):
    shapes = np.zeros((len(subjects), 3), dtype='int')
    for i, subject in enumerate(tqdm(subjects, desc='calculating max shape')):
        shapes[i, :] = subject.img.spatial_shape
    max_shape = tuple(np.max(shapes, axis=0))
    print(f'Largest image shape after cropping & resampling: {max_shape}')
    return max_shape



# ---------------------------------------------- MAIN CLASSES / FUNCTIONS ---------------------------------------------


cg = Config()
save_intermediate_images = cg.save_intermediate_images
root = cg.root
labels_remap_dictionary = cg.labels_remap_dictionary
data_folder = join(cg.root, 'data')
images_folder = join(data_folder, 'images')
results_folder = cg.results_folder
os.makedirs(results_folder, exist_ok=True)
log_file = cg.log_file
subject_names = subdirs(images_folder, join=False)



# --------------------------------------- Preprocessing for Yale Glioma Dataset --------------------------------------


###################
# brain extraction
###################

print_and_log(f'\n   >>>   Brain Extraction   <<<\n', log_file)
t0 = datetime.now()
bet_raw_folder = join(data_folder, 'bet_raw')
os.makedirs(bet_raw_folder, exist_ok=True)
bet_results_folder = join(data_folder, 'bet_results')
os.makedirs(bet_results_folder, exist_ok=True)

for subj in tqdm(subject_names, desc='copying images to bet_raw_folder'):
    shutil.copyfile(join(images_folder, subj, 'img.nii.gz'), join(bet_raw_folder, f'{subj}__img.nii.gz'))

command = f'hd-bet -i {bet_raw_folder} -o {bet_results_folder}'
os.system(command)

shutil.rmtree(bet_raw_folder)

for subj in tqdm(subject_names, desc='moving images from bet_results_folder back to subject folders'):
    shutil.move(join(bet_results_folder, f'{subj}__img.nii.gz'),
                join(images_folder, subj, 'img_bet.nii.gz'))
    shutil.move(join(bet_results_folder, f'{subj}__img_mask.nii.gz'),
                join(images_folder, subj, 'img_bet_mask.nii.gz'))

shutil.rmtree(bet_results_folder)
print_and_log(f'\n   >>>   Brain extraction duration: {datetime.now() - t0}   <<<\n', log_file)





########################
# bias field correction
########################

print_and_log(f'\n   >>>   Correcting N4 Bias Field   <<<\n', log_file)
t0 = datetime.now()
bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()

def correct_bias_field(i):
    subj = subject_names[i]
    nifti_path = join(images_folder, subj, 'img_bet.nii.gz')
    nifti2_path = join(images_folder, subj, 'img_bias_corrected.nii.gz')
    if exists(nifti2_path):
        print_and_log(f'Already done for subject {i}: {subj}',
                      log_file)
        return
    nifti = sitk.ReadImage(nifti_path)
    dtype = sitk.GetArrayFromImage(nifti).dtype
    nifti = sitk.Cast(nifti, sitk.sitkFloat32)
    msk = sitk.OtsuThreshold(nifti, 0, 1, 200)
    nifti = bias_corrector.Execute(nifti, msk)
    img = sitk.GetArrayFromImage(nifti).astype(dtype)
    nifti2 = sitk.GetImageFromArray(img)
    nifti2.CopyInformation(nifti)
    sitk.WriteImage(nifti2, nifti2_path)
    print_and_log(f'Bias field corrected for subject {i}: {subj}   |   time from start: {datetime.now() - t0}',
                  log_file)

if __name__ == '__main__':

    # # Method 1: don't determine number of workers:
    # with Pool() as p:
    #     p.map(correct_bias_field, range(len(subject_names)))

    # Method 2: determine number of workers:
    # n_processors = cpu_count() - 1
    n_processors = 5
    print(f'   >>>   number of parallel processors: {n_processors}   <<<\n')
    with Pool(n_processors) as p:
        p.map(correct_bias_field, range(len(subject_names)))

    print_and_log(f'\n   >>>   Bias field correction duration: {datetime.now() - t0: .0f}   <<<\n', log_file)




#####################################################################
# cropping, normalization, label remapping, resampling, and padding
#####################################################################

normalizer = tio.ZNormalization()
label_remapper = tio.RemapLabels(labels_remap_dictionary)

subjects = []

# subj = subject_names[0]
for subj in tqdm(subject_names, desc='Croppoing, normalizing images, and remapping labels'):
    img = tio.ScalarImage(join(images_folder, subj, 'img_bias_corrected.nii.gz'))
    img_orig = tio.ScalarImage(join(images_folder, subj, 'img.nii.gz'))
    seg = tio.LabelMap(join(images_folder, subj, 'seg.nii.gz'))
    '''
    This is because HD-BET slightly changes the affines from img.nii.gz --> img_bet.nii.gz. Therefore we check if the
    image after BET has an affine that is close to the segmentation affine. Then we assign the segmentation's affine
    to be same of image affine after BET (they're close anyway).
    '''
    assert np.allclose(img.affine, seg.affine, atol=0.0001)
    seg2 = tio.LabelMap(tensor=seg.data, affine=img.affine)
    subject = tio.Subject(img=img, seg=seg2, name=subj)
    bet_mask = tio.LabelMap(join(images_folder, subj, 'img_bet_mask.nii.gz'))

    # Crop:
    pad = 0
    bg = 0
    msk = bet_mask.numpy()[0]
    mskw = np.where(msk != bg)
    x1, x2 = int(np.min(mskw[0])) - pad, msk.shape[0] - int(np.max(mskw[0])) - pad
    y1, y2 = int(np.min(mskw[1])) - pad, msk.shape[1] - int(np.max(mskw[1])) - pad
    z1, z2 = int(np.min(mskw[2])) - pad, msk.shape[2] - int(np.max(mskw[2])) - pad
    cropper = tio.transforms.Crop((x1, x2, y1, y2, z1, z2))
    subject = cropper(subject)
    if save_intermediate_images:
        subject.img.save(join(images_folder, subject.name, 'img_cropped.nii.gz'))
        subject.seg.save(join(images_folder, subject.name, 'seg_cropped.nii.gz'))


    # Normalize image:
    img = normalizer(subject.img)

    # Remap segmentation:
    seg = label_remapper(subject.seg)

    # Construct TorchIO subject
    subject = tio.Subject(img=img, seg=seg, name=subj)
    if save_intermediate_images:
        subject.img.save(join(images_folder, subject.name, 'img_normalized.nii.gz'))
        subject.seg.save(join(images_folder, subject.name, 'seg_remapped.nii.gz'))
    subjects.append(subject)



# Resample:

median_voxsize = calculate_median_voxsize(subjects)
resampler = tio.Resample(median_voxsize)
for i, subject in enumerate(tqdm(subjects, desc='Resampling')):
    # resample:
    subject = resampler(subject)
    if save_intermediate_images:
        subject.img.save(join(images_folder, subject.name, 'img_resampled.nii.gz'))
        subject.seg.save(join(images_folder, subject.name, 'seg_resampled.nii.gz'))
    subjects[i] = subject



# Pad (to make all images the same size)

max_shape = calculate_max_shape(subjects)
padder = tio.CropOrPad(max_shape)
for i, subject in enumerate(tqdm(subjects, 'Padding')):
    subject = padder(subject)
    if save_intermediate_images:
        subject.img.save(join(images_folder, subject.name, 'img_padded.nii.gz'))
        subject.seg.save(join(images_folder, subject.name, 'seg_padded.nii.gz'))
    subjects[i] = subject



# Final checks & save pre-processed images:

# check if all images have the same shape:
img_shapes = np.zeros((len(subjects), 3))
seg_shapes = np.zeros_like(img_shapes)
for i, subject in enumerate(tqdm(subjects, desc='checking image shapes')):
    img_shapes[i, :] = subject.img.spatial_shape
    seg_shapes[i, :] = subject.seg.spatial_shape

print_and_log(f'''

   >>>   Ensuring that all images and segmentations have consistent shapes   <<<

image shapes: {np.unique(img_shapes, axis=0)}
seg shapes: {np.unique(seg_shapes, axis=0)}
''', log_file)

# check if the image & segmentation in each subject are consistent:
for subject in tqdm(subjects, desc='checking img & seg consistency'):
    try:
        subject.check_consistent_affine()
        subject.check_consistent_space()
        subject.check_consistent_spatial_shape()
        subject.check_consistent_orientation()
    except RuntimeError as err:
        print_and_log(f'Inconsistent image and segmentation in subject: {subject.name}',
                      log_file)

# save pre-processed images:
for subject in tqdm(subjects, desc='saving pre-processed images'):
    path = join(images_folder, subject.name, 'img_preprocessed.nii.gz')
    subject.img.save(path)
    path = join(images_folder, subject.name, 'seg_preprocessed.nii.gz')
    subject.seg.save(path)






# ------------------------------------------- Preprocessing for ADNI Dataset -----------------------------------------

subjects = []

# Construct subjects:

# subj = subject_names[0]
for subj in tqdm(subject_names, desc='constructing subjects'):
    img = tio.ScalarImage(join(images_folder, subj, 'img.nii.gz'))
    seg = tio.LabelMap(join(images_folder, subj, 'seg.nii.gz'))
    assert np.allclose(img.affine, seg.affine, atol=0.0001)
    subject = tio.Subject(img=img, seg=seg, name=subj)
    subjects.append(subject)



# Pad (to make all images the same size):

max_shape = calculate_max_shape(subjects)
padder = tio.CropOrPad(max_shape)
for i, subject in enumerate(tqdm(subjects, 'Padding')):
    subject = padder(subject)
    if save_intermediate_images:
        subject.img.save(join(images_folder, subject.name, 'img_padded.nii.gz'))
        subject.seg.save(join(images_folder, subject.name, 'seg_padded.nii.gz'))
    subjects[i] = subject

# Final checks & save pre-processed images:

# check if all images have the same shape:
img_shapes = np.zeros((len(subjects), 3))
seg_shapes = np.zeros_like(img_shapes)
for i, subject in enumerate(tqdm(subjects, desc='checking image shapes')):
    img_shapes[i, :] = subject.img.spatial_shape
    seg_shapes[i, :] = subject.seg.spatial_shape

print_and_log(f'''

   >>>   Ensuring that all images and segmentations have consistent shapes   <<<

image shapes: {np.unique(img_shapes, axis=0)}
seg shapes: {np.unique(seg_shapes, axis=0)}
''', log_file)

# check if the image & segmentation in each subject are consistent:
for subject in tqdm(subjects, desc='checking img & seg consistency'):
    try:
        subject.check_consistent_affine()
        subject.check_consistent_space()
        subject.check_consistent_spatial_shape()
        subject.check_consistent_orientation()
    except RuntimeError as err:
        print_and_log(f'Inconsistent image and segmentation in subject: {subject.name}',
                      log_file)

# save pre-processed images:
for subject in tqdm(subjects, desc='saving pre-processed images'):
    path = join(images_folder, subject.name, 'img_preprocessed.nii.gz')
    subject.img.save(path)
    path = join(images_folder, subject.name, 'seg_preprocessed.nii.gz')
    subject.seg.save(path)

# .....................................................................................................................

# Label remapping:
label_remapping = cg.label_remapping
label_remap_dictionary = cg.labels_remap_dictionary
one_hot_encoding = cg.one_hot_encoding
n_labels = cg.n_labels
print(f'n labels: {n_labels}')

if label_remapping:
    remapper = tio.RemapLabels(remapping=label_remap_dictionary)
if one_hot_encoding:
    one_hot_encoder = tio.OneHot(n_labels)


for subject_name in tqdm(subject_names, desc='remapping & one-hot-encoding'):
    seg = tio.LabelMap(join(images_folder, subject_name, 'seg_preprocessed.nii.gz'))
    if label_remapping:
        seg = remapper(seg)
    if one_hot_encoding:
        seg = one_hot_encoder(seg)
    seg.save(join(images_folder, subject_name, 'seg_prepared.nii.gz'))



# -------------------------------------------------- CODE TESTING -----------------------------------------------------


delete_list = ['img_bet_mask.nii.gz',
               'img_bet.nii.gz',
               'img_bias_corrected.nii.gz']

recursive_delete(images_folder, delete_list)




folder = '/Users/sa936/projects/sccapsnet/data/images/002_S_0295__2006-04-18'
img = tio.ScalarImage(join(folder, 'img.nii.gz'))
img2 = tio.ScalarImage(join(folder, 'img_preprocessed.nii.gz'))
seg2 = tio.LabelMap(join(folder, 'seg_prepared.nii.gz'))
data = seg2.data

imgshow(data)
