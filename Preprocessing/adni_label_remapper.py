
"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This file contains tools to help patch 3D images.
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:
from os_tools import subdirs
from image_tools import imgshow

# System imports:
import numpy as np
import torch
import torchio as tio
import pandas as pd
import os
from os.path import join
import nibabel as nib
from tqdm import tqdm


# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)

# ---------------------------------------------- MAIN CLASSES / FUNCTIONS ---------------------------------------------

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
subjects = subdirs(images_folder)

label, label_name = 53, 'r_hippocampus'

subject = subjects[0]
for subject in tqdm(subjects, desc=f'generating {label_name} binary labels'):
    seg_path = join(images_folder, subject, 'seg_preprocessed.nii.gz')
    seg = nib.load(seg_path)
    segd = seg.get_fdata()
    segd2 = np.zeros_like(segd)
    segd2[segd == label] = 1
    seg2 = nib.Nifti1Image(segd2, seg.affine)
    seg_path2 = join(images_folder, subject, f'{label_name}.nii.gz')
    nib.save(seg2, seg_path2)







# ------------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------



# .....................................................................................................................





# -------------------------------------------------- CODE TESTING -----------------------------------------------------

