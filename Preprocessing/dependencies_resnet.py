#!/usr/bin/python
# coding: utf-8


import os
import io
from io import BytesIO
import sys
import json
import glob
import random
import collections
import time
import re
# import requests
import zipfile
import statistics
import logging
import numpy as np
import pandas as pd
# import pydicom
# from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
# import seaborn as sns
import copy
import tempfile
from typing import Optional, Any, Mapping, Hashable, DefaultDict, Tuple, List
from tqdm import tqdm
import collections
from functools import partial

import torch

torch.cuda.empty_cache()
from torch import nn, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Sigmoid
from torch.utils import data as torch_data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.optim import Adam

from efficientnet_pytorch_3d import EfficientNet3D
from sklearn import model_selection as sk_model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, PassiveAggressiveRegressor, LinearRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.utils import shuffle
import skimage.transform as skTrans
from skimage.exposure import rescale_intensity
import scipy.ndimage
from scipy.stats import zscore
from IPython.display import Image
import SimpleITK as sitk
import nibabel as nib
from nilearn.image.image import _crop_img_to as crop_img_to
from nilearn.image.image import check_niimg
from nilearn.image import new_img_like
# !nvidia-smi
# !pip install -qU "monai[ignite, nibabel, torchvision, tqdm]==0.6.0"
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import urllib
from urllib.request import urlopen
import pydicom
from pydicom.pixel_data_handlers import apply_voi_lut

from medcam import medcam
import monai
from monai.data import ImageDataset, ArrayDataset, CacheDataset, create_test_image_3d, pad_list_data_collate, Dataset, \
    DataLoader
from monai.visualize import CAM, GradCAM
from monai.config import print_config, KeysCollection
from monai.utils import first
from monai.transforms import (
    Transform,
    MapTransform,
    Randomizable,
    AddChannel,
    AddChanneld,
    AsChannelFirst,
    AsChannelFirstd,
    Compose,
    LoadImage,
    SaveImage,
    LoadImaged,
    SaveImaged,
    Lambda,
    Lambdad,
    CenterSpatialCrop,
    RandSpatialCrop,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    ToTensor,
    ToTensord,
    Orientation,
    Rotate,
    Rotated,
    RandRotate90,
    Resize,
    Resized,
    ScaleIntensity,
    ScaleIntensityd,
    NormalizeIntensity,
    Spacing,
    Spacingd,
    EnsureType
)
from monai.transforms import (
    Orientation,
    RandAffine,
    Rand3DElastic,
    RandRotate,
    RandFlip,
    RandZoom
)

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.inferers import SlidingWindowInferer
from monai.metrics import compute_meandice


# print_config()

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60

    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


