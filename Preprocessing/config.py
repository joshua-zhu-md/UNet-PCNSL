"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This module contains the Config class to set configurations for the project.

Reminders:

1. disabled autocast
2. decreased size of patch to 64


"""


# ---------------------------------------------------- Imports -------------------------------------------------------

# Project imports:
# import os
# if '/home/arman_avesta/' in os.getcwd():
#     from nncapsnet.models import UNet3D
#     from nncapsnet.loss_functions import DiceLoss, DiceBCELoss
#     from nncapsnet.os_tools import convert_csv_to_list, subdirs
# else:
#     from models import UNet3D
#     from loss_functions import DiceLoss, DiceBCELoss


from models import UNet3D, CapsNet3D
from loss_functions import DiceLoss, DiceBCELoss
from os_tools import convert_csv_to_list, subdirs



# System imports:
import numpy as np
import torch
import torchio as tio
import os
from os.path import join
from datetime import datetime
from tqdm import tqdm
from multiprocessing import cpu_count


# ------------------------------------------------- config class -----------------------------------------------------

class Config:

    def __init__(self):
        ###########################################################
        #                  SET TRAINING PARAMETERS                #
        ###########################################################

        # Experiment notes (to be saved):
        self.experiment = 'capsu1'

        # Set project root path:
        # self.root = '/home/arman_avesta/nncapsnet'
        path = os.getcwd()
        if '/Users/josh/' in path:
            self.root = '/Users/josh/Documents/Aneja Lab/PCNSL/'
        elif '/home/joshua_zhu/' in path:
            self.root = '/home/joshua_zhu/PCNSL'
        else:
            self.root = '/home/joshua_zhu/toydataset'
            # raise Exception('edit config.py --> set self.root')

        # Folder to save model results:

        self.train_csv = join(self.root, 'data', 'csvs', 'train_subjects.csv')
        self.valid_csv = join(self.root, 'data', 'csvs', 'valid_subjects.csv')
        self.test_csv = join(self.root, 'data', 'csvs', 'test_subjects.csv')


        now = datetime.now().strftime('%m_%d_%Y')
        self.results_folder = join(self.root, 'data', 'results', f'{self.experiment}_{now}')

        self.log_file = join(self.results_folder, 'log_file.txt')


        # Determine if backup to S3 should be done:
        self.s3backup = False
        # S3 bucket backup folder for results:
        self.s3_results_folder = join('s3://aneja-lab-sccapsnet', 'results', f'{self.experiment}_{now}')

        self.images_folder = join(self.root, 'data', 'images')

        # Set model:
        self.model = UNet3D(do_sigmoid=False)

        # Set loss function: options are DiceLoss and DiceBCELoss
        # self.loss_function = DiceLoss(conversion='margin')
        # self.validation_loss_function = DiceLoss(conversion='threshold')

        self.loss_function = DiceBCELoss()
        self.validation_loss_function = DiceBCELoss()

        # Set number of training epochs:
        self.n_epochs = 1000
        # Set training batch size:
        self.train_batch_size = 4
        # Set validation batch size:
        self.valid_batch_size = 8
        # Set the patch size:
        # if this is set to 100, the patch size will be 100 x 100 x 100.
        # note that (x, y, z) here respectively represent left-right, posterior-anterior, and inferior-superior
        # dimensions in the standard radiology coordinate system ('L','A','S').
        # self.patch_size = (128, 128, 32)
        self.patch_size = (64, 64, 64)

        # Set the patch sw_overlap size (in relation to patch shape):
        self.patch_overlap = (30, 30, 30)


        # Torchio data loader parameters:
        # self.num_workers = cpu_count() - 1
        self.num_workers = 4
        self.queue_length = 100
        self.samples_per_volume = 10

        # label remapping:
        # self.label_remapping = True

        self.label_remapping = False                            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        # yale glioma remap dict:
        self.labels_remap_dictionary = {0:0, 255:1}

        # adni remap dictionary:
        # self.labels_remap_dictionary = {0: 0, 2: 0, 4: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
        #                                 15: 0, 16: 0, 17: 0, 18: 0, 24: 0, 26: 0, 28: 0, 29: 0, 30: 0, 31: 0,
        #                                 41: 0, 43: 0, 44: 0, 46: 0, 47: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 1,
        #                                 54: 0, 58: 0, 60: 0, 62: 0, 63: 0, 72: 0, 77: 0, 80: 0, 85: 0, 251: 0,
        #                                 252: 0, 253: 0, 254: 0, 255: 0, 1000: 0, 1001: 0, 1002: 0, 1003: 0,
        #                                 1004: 0, 1005: 0, 1006: 0, 1007: 0, 1008: 0, 1009: 0, 1010: 0, 1011: 0,
        #                                 1012: 0, 1013: 0, 1014: 0, 1015: 0, 1016: 0, 1017: 0, 1018: 0, 1019: 0,
        #                                 1020: 0, 1021: 0, 1022: 0, 1023: 0, 1024: 0, 1025: 0, 1026: 0, 1027: 0,
        #                                 1028: 0, 1029: 0, 1030: 0, 1031: 0, 1032: 0, 1033: 0, 1034: 0, 1035: 0,
        #                                 2000: 0, 2001: 0, 2002: 0, 2003: 0, 2004: 0, 2005: 0, 2006: 0, 2007: 0,
        #                                 2008: 0, 2009: 0, 2010: 0, 2011: 0, 2012: 0, 2013: 0, 2014: 0, 2015: 0,
        #                                 2016: 0, 2017: 0, 2018: 0, 2019: 0, 2020: 0, 2021: 0, 2022: 0, 2023: 0,
        #                                 2024: 0, 2025: 0, 2026: 0, 2027: 0, 2028: 0, 2029: 0, 2030: 0, 2031: 0,
        #                                 2032: 0, 2033: 0, 2034: 0, 2035: 0}

        '''
        0: background
        17 --> 0: left hippocampus
        53 --> 1: right hippocampus
        10 --> 0: left thalamus
        49 --> 0: right thalamus
        14 --> 0: 3rd ventricle
        '''

        '''
        0: background
        17 --> 1: left hippocampus
        53 --> 2: right hippocampus
        10 --> 3: left thalamus
        49 --> 4: right thalamus
        14 --> 5: 3rd ventricle
        '''


        self.one_hot_encoding = False
        self.n_labels = len(np.unique(list(self.labels_remap_dictionary.values())))


        # determine train sampler type; options: 'UniformSampler' or 'LabelSampler':
        self.train_sampler_type = 'LabelSampler'

        # in case sampler type is LabelSampler, set the probabilities:
        self.labels_patch_sample_probabilities = {0: 0.5, 1: 0.5}
        # self.labels_patch_sample_probabilities = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1}

        # save intermediate images during pre-processing:
        self.save_intermediate_images = True


        # Set initial learning rate:
        self.lr_initial = 0.002
        # don't decrease learning rate lower than this minimum:
        self.lr_min = 0.0001
        # set the factor by which the learning rate will be reduced:
        self.lr_factor = 0.5
        # number of validation epochs without loss improvement before learning rate is decreased;
        # if patience = 4 --> optimizer decreases learning rate after 5 validation epochs without loss improvement:
        self.lr_patience = 5
        # ignore validation loss changes smaller than this threshold:
        self.lr_loss_threshold = 0.001
        self.lr_threshold_mode = 'abs'
        # Set optimizer: default is Adam optimizer:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_initial)

        # Set the loss threshold to update best model
        self.best_loss_threshold = self.lr_loss_threshold

        # .......................................................................................................


        self.train_names = convert_csv_to_list(self.train_csv)
        self.valid_names = convert_csv_to_list(self.valid_csv)
        self.test_names = convert_csv_to_list(self.test_csv)

        # subjects = []
        # for subject_name in self.train_names:
        #     subject = tio.Subject(img=tio.ScalarImage(join(self.images_folder, subject_name,
        #                                                    'img_preprocessed.nii.gz')))
        #     subjects.append(subject)
        #
        # max_img_shape = calculate_max_shape(subjects)
        # patch_size, patch_overlap = np.array(self.patch_size), np.array(self.patch_overlap)
        # self.patch_size = tuple(np.minimum(patch_size, max_img_shape))
        # # self.patch_overlap = tuple(np.round(self.patch_size * patch_overlap).astype('int'))




# ------------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------

def calculate_max_shape(subjects):
    shapes = np.zeros((len(subjects), 3), dtype='int')
    for i, subject in enumerate(tqdm(subjects, desc='calculating max shape')):
        shapes[i, :] = subject.img.spatial_shape
    max_shape = np.max(shapes, axis=0)
    print(f'Largest image shape after pre-processing: {max_shape}')
    return max_shape


# -------------------------------------------------- CODE TESTING -----------------------------------------------------

if __name__ == '__main__':

    cg = Config()

    print(f'''
    cg.root: {cg.root}
    cg.results_folder: {cg.results_folder}
    cg.s3_results_folder: {cg.s3_results_folder}
    patch overlap: {cg.patch_overlap}
    ''')

