"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/15/2022)

This file contains tools that help in image visualization, processing, training, and evaluation..
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:


# System imports:
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)


# ------------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------

def reorient_nifti(nifti):
    """
    This function re-orients the MRI volume into the standard radiology system, i.e. ('L','A','S') = LAS+,
    and also corrects the affine transform for the volume (from the MRI volume space to the scanner space).
    Input:
        - nifti: MRI volume NIfTI file
    Outputs:
        - corrected nifti: corrected MRI volume in the standard radiology coordinate system: ('L','A','S') = LAS+.
    Notes:
    ------
    nib.io_orientation compares the orientation of nifti with RAS+ system. So if nifti is already in
    RAS+ system, the return from nib.io_orientation(nifti.affine) will be:
    [[0, 1],
     [1, 1],
     [2, 1]]
    If nifti is in LAS+ system, the return would be:
    [[0, -1],           # -1 means that the first axis is flipped compared to RAS+ system.
     [1, 1],
     [2, 1]]
    If nifti is in PIL+ system, the return would be:
    [[1, -1],           # P is the 2nd axis in RAS+ hence 1 (not 0), and is also flipped hence -1.
     [2, -1],           # I is the 3rd axis in RAS+ hence 2, and is also flipped hence -1.
     [0, -1]]           # L is the 1st axis in RAS+ hence 0, and is also flipped hence -1.
    Because we want to save images in LAS+ orientation rather than RAS+, in the code below we find axis 0 and
    negate the 2nd colum, hence going from RAS+ to LAS+. For instance, for PIL+, the orientation will be:
    [[1, -1],
     [2, -1],
     [0, -1]]
    This is PIL+ compared to RAS+. To compare it to LAS+, we should change it to:
    [[1, -1],
     [2, -1],
     [0, 1]]
    That is what this part of the code does:
    orientation[orientation[:, 0] == 0, 1] = - orientation[orientation[:, 0] == 0, 1]
    Another inefficient way of implementing this function is:
    ################################################################################
    original_orientation = nib.io_orientation(nifti.affine)
    target_orientation = nib.axcodes2ornt(('L', 'A', 'S'))
    orientation_transform = nib.ornt_transform(original_orientation, target_orientation)
    return nifti.as_reoriented(orientation_transform)
    ################################################################################
    """
    orientation = nib.io_orientation(nifti.affine)
    orientation[orientation[:, 0] == 0, 1] = - orientation[orientation[:, 0] == 0, 1]
    return nifti.as_reoriented(orientation)

# .....................................................................................................................

def reoirent_img(img, voxsize=(1,1,1), coords=('L','A','S')):
    """
    This function re-orients a 3D ndarray volume into the standard radiology coordinate system: ('L','A','S') = LAS+.

    :param img: {ndarray}
    :param voxsize: {tuple, list, or ndarray}
    :param coords: {tuple or list}: coordinate system of the image. e.g. ('L','P','S').
        for more info, refer to: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm

    :return: reoriented image and voxel size in standard radiology coordinate system ('L','A','S) = LAS+ system.
    """
    assert img.ndim == 3, 'image should have shape (x, y, z)'
    if coords == ('L', 'A', 'S'):
        return img, voxsize
    if coords == ('R', 'A', 'S'):
        img = np.flip(img, 0)
        return img, voxsize
    if coords == ('R', 'P', 'I'):
        # R,P,I --> L,A,S
        img = np.flip(img)              # flip the image in all 3 dimensions
        return img, voxsize
    if coords == ('P', 'I', 'R'):
        # P,I,R --> A,S,L
        img = np.flip(img)
        # A,S,L --> L,A,S
        img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
        voxsize = (voxsize[1], voxsize[2], voxsize[0])
        return img, voxsize
    if coords == ('R', 'I', 'A'):
        # R,I,A --> R,A,I:
        img = np.swapaxes(img, 1, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        # R,A,I --> L,A,S:
        img = np.flip(img, [0, 2])
        return img, voxsize
    if coords == ('L', 'I', 'P'):
        # L,I,P --> L,P,I
        img = np.swapaxes(img, 1, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        # L,P,I --> L,A,S
        img = np.flip(img, [1, 2])
        return img, voxsize
    if coords == ('P', 'R', 'S'):
        # P,R,S --> R,P,S:
        img = np.swapaxes(img, 0, 1)
        voxsize = (voxsize[1], voxsize[0], voxsize[2])
        # R,P,S --> L,A,S
        img = np.flip(img, [0, 1])
        return img, voxsize
    if coords == ('L', 'I', 'A'):
        img = np.swapaxes(img, 1, 2)
        img = np.flip(img, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        return img, voxsize
    if coords == ('L', 'S', 'A'):
        img = np.swapaxes(img, 1, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        return img, voxsize
    if coords == ('L', 'P', 'S'):
        img = np.flip(img, 1)
        return img, voxsize
    raise Exception('coords not identified: please revise the imshow function and define the coordinate system')

# .....................................................................................................................

def imgshow(img, voxsize=(1,1,1), coords=('L','A','S')):
    """
    This function shows 2D/3D/4D/5D images & image batches:
    - 2D: image is shown.
    - 3D: the volume mid-slices in axial, coronal and sagittal planes are shown.
    - 4D: assumes that image is multichannel image (channel-first) and shows all channels as 3D images.
    - 5D: assumes that image is a batch of multichannel images (batch first, channel second), and shows all
        batches & all channels of 3D images.

    :param img: {ndarray, tensor, or nifti}
    :param voxsize: {tuple, list, or ndarray} Default=(1,1,1)
    :param coords: image coordinate system; Default=('L','A','S') which is the standard radiology coordinate system.
        for more info, refer to: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm

    :return: None. Side effect: shows image(s).
    """
    if type(img) is nib.nifti1.Nifti1Image:
        voxsize = img.header.get_zooms()
        coords = nib.aff2axcodes(img.affine)
        img = img.get_fdata()
    elif type(img) is torch.Tensor:
        img = img.numpy()

    kwargs = dict(cmap='gray', origin='lower')
    ndim = img.ndim
    assert ndim in (2, 3, 4 ,5), f'image shape: {img.shape}; imshow can only show 2D and 3D images, ' \
                                 f'multi-channel 3D images (4D), and batches of multi-channel 3D images (5D).'

    if ndim == 2:
        plt.imshow(img.T, **kwargs)
        plt.show()

    elif ndim == 3:
        img, voxsize = reoirent_img(img, voxsize, coords)
        midaxial = img.shape[2] // 2
        midcoronal = img.shape[1] // 2
        midsagittal = img.shape[0] // 2
        axial_aspect_ratio = voxsize[1] / voxsize[0]
        coronal_aspect_ratio = voxsize[2] / voxsize[0]
        sagittal_aspect_ratio = voxsize[2] / voxsize[1]

        axial = plt.subplot(2, 3, 1)
        plt.imshow(img[:, :, midaxial].T, **kwargs)
        axial.set_aspect(axial_aspect_ratio)

        coronal = plt.subplot(2, 3, 2)
        plt.imshow(img[:, midcoronal, :].T, **kwargs)
        coronal.set_aspect(coronal_aspect_ratio)

        sagittal = plt.subplot(2, 3, 3)
        plt.imshow(img[midsagittal, :, :].T, **kwargs)
        sagittal.set_aspect(sagittal_aspect_ratio)

        plt.show()

    elif ndim in (4, 5):
        for i in range(img.shape[0]):
            imgshow(img[i, ...])

# .....................................................................................................................

def crop_nifti(nifti, msk, pad=1, bg=0):
    """
    This function crops a NIfTI file according to the mask.
    :param nifti: nifti image
    :param msk: nifti mask
    :param pad: (int) number of voxels to pad around the cropped image, default=1.
    :param bg: (int) background value, default=0.
    :return:
    """
    msk = np.where(msk != bg)
    minx, maxx = int(np.min(msk[0])) - pad, int(np.max(msk[0])) + pad + 1
    miny, maxy = int(np.min(msk[1])) - pad, int(np.max(msk[1])) + pad + 1
    minz, maxz = int(np.min(msk[2])) - pad, int(np.max(msk[2])) + pad + 1
    '''
    # The longer way of implementing this:
    new_origin_img_space = np.array([minx, miny, minz])
    new_origin_scanner_space = nib.affines.apply_affine(nifti.affine, new_origin_img_space)
    new_affine = nifti.affine.copy()
    new_affine[:3, 3] = new_origin_scanner_space
    img = nifti.get_fdata()
    img_cropped = img[minx:maxx, miny:maxy, minz:maxz]
    nifti_cropped = nib.Nifti1Image(img_cropped, new_affine)
    return nifti_cropped
    '''
    return nifti.slicer[minx:maxx, miny:maxy, minz:maxz]



# -------------------------------------------------- CODE TESTING -----------------------------------------------------

if __name__ == '__main__':
    print(f'''
    
    ''')




