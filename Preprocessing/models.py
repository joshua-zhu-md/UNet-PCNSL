"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This file contains model classes.
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:


# System imports:
import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad, softmax

# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)

# ---------------------------------------------- MAIN CLASSES / FUNCTIONS ---------------------------------------------

class UNet3D(nn.Module):
    """
    This class defines the architecture of the 3D UNet
    """

    def __init__(self, Ci=1, Co=1, xpad=True, do_sigmoid=False):
        """
        Inputs:
            - Ci: number of input channels into the 3D UNet
            - Co: number of output channels logits_to_preds of the 3D UNet
            - xpad: if the input size is not 2^n in all dimensions, set xpad to True.
            - sig: apply sigmoid function in the last layer
        """
        super().__init__()

        # Downsampling limb of UNet:
        self.left1 = Doubleconv(Ci, 64)
        self.left2 = DownDoubleconv(64, 128)
        self.left3 = DownDoubleconv(128, 256)
        self.left4 = DownDoubleconv(256, 512)

        # Bottleneck of UNet:
        self.bottom = DownDoubleconv(512, 1024)

        # Upsampling limb of UNet:
        # the units are numbered in reverse to match the corresponding downsampling units
        self.right4 = UpConcatDoubleconv(1024, 512, xpad)
        self.right3 = UpConcatDoubleconv(512, 256, xpad)
        self.right2 = UpConcatDoubleconv(256, 128, xpad)
        self.right1 = UpConcatDoubleconv(128, 64, xpad)

        self.out = Outconv(64, Co, do_sigmoid=do_sigmoid)

    def forward(self, x):
        """
        Input:
            - x: UNet input; type: torch tensor; dimensions: [B, Ci, D, H, W]
        Output:
            - UNet output; type: torch tensor; dimensions: output[B, Co, D, H, W]
        Dimensions explained:
            - 'i' and 'o' subscripts repsectively mean inputs and outputs.
            - C: channels
            - D: depth
            - H: height
            - W: width
        """
        # Downsampling limb of UNet:
        x1 = self.left1(x)
        x2 = self.left2(x1)
        x3 = self.left3(x2)
        x4 = self.left4(x3)

        # Bottleneck of UNet:
        x = self.bottom(x4)

        # Upsampling limb of UNet:
        x = self.right4(x4, x)
        x = self.right3(x3, x)
        x = self.right2(x2, x)
        x = self.right1(x1, x)

        return self.out(x)

# .....................................................................................................................


# --------------------------------------------- HELPER FUNCTIONS / CLASSES --------------------------------------------

class Doubleconv(nn.Module):
    """
    DoubleConvolution units in the 3D UNet
    """

    def __init__(self, Ci, Co):
        """
        Inputs:
            - Ci: number of input channels into the DoubleConvolution unit
            - Co: number of output channels logits_to_preds of the DoubleConvolution unit
        """
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv3d(Ci, Co, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(Co),
            nn.ReLU(inplace=True),
            nn.Conv3d(Co, Co, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(Co),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Input:
        - x: torch tensor; dimensions: [B, Ci, Di, Hi, Wi]
        Output:
             - return: x --> conv3d --> batch_norm --> ReLU --> conv3d --> batch_norm --> ReLU --> output
                dimensions: [B, Co, Do, Ho, Wo]
        """
        return self.doubleconv(x)

# .....................................................................................................................

class DownDoubleconv(nn.Module):
    """
    Units in the left side of the 3D UNet:
    Down-sample using MaxPool3d --> then DoubleConvolution
    """

    def __init__(self, in_ch, out_ch):
        """
        Inputs:
            - Ci: number of input channels into the DownDoubleconv unit
            - Co: number of output channels logits_to_preds of the DownDoubleconv unit
        """
        super().__init__()
        self.maxpool_doubleconv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            Doubleconv(in_ch, out_ch))


    def forward(self, x):
        """
        Input:
            - x: torch tensor; dimensions: x[batch, channels, D, H, W]
        Output:
            - return: x --> maxpool3d --> DoubleConv Unit --> output
        """
        return self.maxpool_doubleconv(x)

# .....................................................................................................................

class UpConcatDoubleconv(nn.Module):
    """
    Units in the right side of the 3D UNet:
    Up-scale using ConvTranspose3d --> Concatenate the bottom and horizontal channels --> DoubleConvolution
    """

    def __init__(self, Ci, Co, xpad=True, up_mode='transposed'):
        """
        Inputs:
            - Ci: number of input channels into the UpConcatDoubleconv unit
            - Co: number of output channels logits_to_preds of the UpConcatDoubleconv unit
            - xpad: set this to False only if the input D/H/W dimensions are all powers of two. Otherwise set
                            this to True.
            - up_mode: default is 'transposed'. Set this to 'trilinear' if you want trilinear interpolation
                            instead (but interpolation would make the network slow).
        """
        super().__init__()
        self.xpad = xpad
        self.up_mode = up_mode

        if self.up_mode == 'transposed':
            self.up = nn.ConvTranspose3d(Ci, Co, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=self.up_mode, align_corners=True)

        self.doubleconv = Doubleconv(Ci, Co)


    def forward(self, x1, x2):
        """
        Inputs:
            - x1: skip-connection from the downsampling limb of U-Net; dimensions: [B, C1, D1, H1, W1]
            - x2: from the lower-level upsampling limb of U-Net; dimensions: [B, C2, D2, H2, W2]
        Output:
            - return: up-scale x2 --> concatenate(x1, x2) --> DoubleConv Unit --> output
        """
        x2 = self.up(x2)

        if self.xpad:
            # If H2 or W2 are smaller than H1 or W1, pad x2 so that its size matches x1.
            B1, C1, D1, H1, W1 = x1.shape
            B2, C2, D2, H2, W2 = x2.shape
            assert B1 == B2
            diffD, diffH, diffW = D1 - D2, H1 - H2, W1 - W2
            x2 = pad(x2, [diffW // 2, diffW - diffW // 2,
                          diffH // 2, diffH - diffH // 2,
                          diffD // 2, diffD - diffD // 2])

        # Concatenate x1 and x2:
        x = torch.cat([x1, x2], dim=1)

        # Return double convolution of the concatenated tensor:
        return self.doubleconv(x)

# .....................................................................................................................

class Outconv(nn.Module):
    """
    Output unit in the 3D UNet
    """

    def __init__(self, Ci, Co, do_sigmoid=False):
        """
        Inputs:
            - in_ch: number of input channels into the final output unit
            - out_ch: number of output channels logits_to_preds of the entire UNet
            - sig: apply sigmoid function in the last layer
        """
        super().__init__()

        if not do_sigmoid:
            self.conv_out = nn.Conv3d(Ci, Co, kernel_size=1)

        else:
            if Co == 1:
                self.conv_out = nn.Sequential(
                    nn.Conv3d(Ci, Co, kernel_size=1),
                    nn.Sigmoid())
            elif Co > 1:
                self.conv_out = nn.Sequential(
                    nn.Conv3d(Ci, Co, kernel_size=1),
                    nn.Softmax(dim=1))
    def forward(self, x):
        return self.conv_out(x)









####################################################################################################################
# ----------------------------------------------- 3D CapsNet model -------------------------------------------------
####################################################################################################################

class CapsNet3D(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, xpad=True):
        """
        Inputs:
        - in_ch: input channels
        - out_ch: output channels
        - xpad: set to True if input shape is not powers of 2.
        Dimensions explained:
        - 'i' and 'o' subscripts respectively represent inputs and outputs.
            For instance, Ci represents number of input capsule channels.
        - C: capsule channels = capsule types.
        - P: pose components = number of elements in the pose vector.
        - K: kernel size for convolutions.
        """
        super().__init__()
        self.xpad = xpad

        self.Conv1 = Conv(in_ch, Po=16, K=5, stride=1, padding=2)
        self.PrimaryCaps2 = ConvCaps(Ci=1, Pi=16, Co=2, Po=16, K=5, stride=2, padding=2, routings=1)
        self.ConvCaps3 = ConvCaps(Ci=2, Pi=16, Co=4, Po=16, K=5, stride=1, padding=2, routings=3)
        self.ConvCaps4 = ConvCaps(Ci=4, Pi=16, Co=4, Po=32, K=5, stride=2, padding=2, routings=3)
        self.ConvCaps5 = ConvCaps(Ci=4, Pi=32, Co=8, Po=32, K=5, stride=1, padding=2, routings=3)
        self.ConvCaps6 = ConvCaps(Ci=8, Pi=32, Co=8, Po=64, K=5, stride=2, padding=2, routings=3)
        self.ConvCaps7 = ConvCaps(Ci=8, Pi=64, Co=8, Po=32, K=5, stride=1, padding=2, routings=3)
        self.DeconvCaps8 = DeconvCaps(Ci=8, Pi=32, Co=8, Po=32, K=4, stride=2, routings=3)
        self.ConvCaps9 = ConvCaps(Ci=16, Pi=32, Co=8, Po=32, K=5, stride=1, padding=2, routings=3)
        self.DeconvCaps10 = DeconvCaps(Ci=8, Pi=32, Co=4, Po=16, K=4, stride=2, routings=3)
        self.ConvCaps11 = ConvCaps(Ci=8, Pi=16, Co=4, Po=16, K=5, stride=1, padding=2, routings=3)
        self.DeconvCaps12 = DeconvCaps(Ci=4, Pi=16, Co=2, Po=16, K=4, stride=2, routings=3)
        self.FinalCaps13 = ConvCaps(Ci=3, Pi=16, Co=1, Po=16, K=1, stride=1, padding=0, routings=3)


    def forward(self, x):
        """
        Inputs:
        - x: batch of input images
        - y: batch of target images
        Outputs:
        - out_seg: segmented image
        - out_recon: reconstructed image within the target mask
        """
        Conv1 = self.Conv1(x)
        PrimaryCaps2 = self.PrimaryCaps2(Conv1)
        ConvCaps3 = self.ConvCaps3(PrimaryCaps2)
        ConvCaps4 = self.ConvCaps4(ConvCaps3)
        ConvCaps5 = self.ConvCaps5(ConvCaps4)
        ConvCaps6 = self.ConvCaps6(ConvCaps5)
        ConvCaps7 = self.ConvCaps7(ConvCaps6)
        DeconvCaps8 = self.concat(ConvCaps5, self.DeconvCaps8(ConvCaps7))
        ConvCaps9 = self.ConvCaps9(DeconvCaps8)
        DeconvCaps10 = self.concat(ConvCaps3, self.DeconvCaps10(ConvCaps9))
        ConvCaps11 = self.ConvCaps11(DeconvCaps10)
        DeconvCaps12 = self.concat(Conv1, self.DeconvCaps12(ConvCaps11))
        FincalCaps13 = self.FinalCaps13(DeconvCaps12)
        SegmentedVolume = vector_norm(FincalCaps13)
        return SegmentedVolume


    def concat(self, skip, x):
        """
        Concatenates two batches of capsules.
        Inputs:
        - skip: skip-connection from the downsampling limb of CapsNet
        - x: input from the upsampling limb of U-Net
        Outputs:
        - concatenated tensor
        Dimensions explained:
        - 's' and 'x' subscripts respectively mean skip-connection input and input from the upsampling limb.
        - B: batch size
        - C: capsule channels = capsule types
        - P: pose components
        - D: depth
        - H: height
        - W: width
        """
        Bs, Cs, Ps, Ds, Hs, Ws = skip.shape
        Bx, Cx, Px, Dx, Hx, Wx = x.shape
        assert (Bs, Ps) == (Bx, Px)
        if self.xpad:
            diffD, diffH, diffW = Ds - Dx, Hs - Hx, Ws - Wx
            x = pad(x, [diffW // 2, diffW - diffW // 2,
                          diffH // 2, diffH - diffH // 2,
                          diffD // 2, diffD - diffD // 2])
        return torch.cat([skip, x], dim=1)



# -------------------------------------------------- 3D CapsNet units -----------------------------------------------


class Conv(nn.Module):
    """
    Non-capsule convolutional layer.
    """
    def __init__(self, in_ch, Po, K, stride, padding=None):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv3d(in_ch, Po, K, stride, padding),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Input:
        - x: MRI images: [B, 1, Di, Hi, Wi]
        Output:
        - x: [B, 1, Po, Do, Ho, Wo]
        """
        x = self.conv(x)                                                    # x: [B, Po, Do, Ho, Wo]
        return x.unsqueeze(1)                                               # return: [B, 1, Po, Do, Ho, Wo]

# ........................................................................................................

class ConvCaps(nn.Module):
    """
    Convolutional capsule layer.
    """
    def __init__(self, Ci, Pi, Co, Po, K, stride, padding, routings=3):
        """
        Inputs:
        - Ci: input capsule channels
        - Pi: input pose components
        - Co: output capsule channels
        - Po: output pose components
        - K: kernel size
        - stride
        - padding
        - routings: dynamic routing iterations
        """
        super().__init__()

        self.Ci = Ci
        self.Pi = Pi
        self.Co = Co
        self.Po = Po
        self.routings = routings

        self.conv = nn.Conv3d(Pi, Co*Po, kernel_size=K, stride=stride, padding=padding, bias=False)
        self.biases = nn.Parameter(torch.zeros(1, 1, Co, Po, 1, 1, 1) + 0.1)    # biases: [1, 1, Co, Po, 1, 1, 1]

    def forward(self, x):                                                       # x: [B, Ci, Pi, Di, Hi, Wi]
        """
        Input:
        - x: batch of input capsules; dimensions: [B, Ci, Pi, Di, Hi, Wi]
        Output:
        - return: batch of output capsules; dimensions: [B, Co, Po, Do, Ho, Wo]
        Dimensions explained:
        - B: batch size
        - C: capsule channels = capsule types
        - P: pose components
        - D: depth
        - H: height
        - W: width
        - 'i' and 'o' subscripts respectively represent input and output dimensions.
            For instance, Ci represents the number of input capsule channels,
            Po represents the number of output pose components in each output capsule,
            and Ho represents the height of the output image.
        """
        B, Ci, Pi, Di, Hi, Wi = x.shape
        assert (Ci, Pi) == (self.Ci, self.Pi)
        x = x.reshape(B*Ci, Pi, Di, Hi, Wi)                                 # x: [B*Ci, Pi, Di, Hi, Wi]
        x = self.conv(x)
        B_Ci, Co_Po, Do, Ho, Wo = x.shape                                   # x: [B*Ci, Co*Po, Do, Ho, Do]
        assert (B_Ci, Co_Po) == (B*Ci, self.Co*self.Po)
        x = x.reshape(B, Ci, self.Co, self.Po, Do, Ho, Wo)                  # x: [B, Ci, Co, Po, Do, Ho, Wo]
        return dynamic_routing(x, self.biases, self.routings)               # return: [B, Co, Po, Do, Ho, Wo]

# ........................................................................................................

class DeconvCaps(nn.Module):
    """
    Transposed convolutional capsule layer (in the upsampling limb of CapsNet)
    """
    def __init__(self, Ci, Pi, Co, Po, K, stride, routings=3):
        super().__init__()

        self.Ci = Ci
        self.Pi = Pi
        self.Co = Co
        self.Po = Po
        self.routings = routings

        self.conv = nn.ConvTranspose3d(Pi, Co*Po, kernel_size=K, stride=stride, bias=False)
        self.biases = nn.Parameter(torch.zeros(1, 1, Co, Po, 1, 1, 1) + 0.1)    # biases: [1, 1, Co, Po, 1, 1, 1]


    def forward(self, x):                                                   # x: [B, Ci, Pi, Di, Hi, Wi]
        B, Ci, Pi, Di, Hi, Wi = x.shape
        assert (Ci, Pi) == (self.Ci, self.Pi)
        x = x.reshape(B*Ci, Pi, Di, Hi, Wi)                                 # x: [B*Ci, Pi, Di, Hi, Wi]
        x = self.conv(x)
        B_Ci, Co_Po, Do, Ho, Wo = x.shape                                   # x: [B*Ci, Co*Po, Do, Ho, Wo]
        assert (B_Ci, Co_Po) == (B*Ci, self.Co*self.Po)
        x = x.reshape(B, Ci, self.Co, self.Po, Do, Ho, Wo)                  # x: [B, Ci, Co, Po, Do, Ho, Wo]
        return dynamic_routing(x, self.biases, self.routings)

# ........................................................................................................

class Decoder(nn.Module):
    """
    To be used for reconstruction loss calculation.
    """
    def __init__(self, Pi=16):
        super().__init__()

        self.recon = nn.Sequential(
            nn.Conv3d(Pi, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True))

    def forward(self, x, y):
        """
        Inputs:
            - x: [B, 1, P, D, H, W]
            - y: [B, 1, D, H, W]
        Output:
            - masked x based on y: [B, P, D, H, W]
        """
        if y is None:
            y = self.create_mask(self.x)
        Bx, Cx, Px, Dx, Hx, Wx = x.shape
        By, Cy, Dy, Hy, Wy = y.shape
        assert (Cx, Cy) == (1, 1)
        assert (Bx, Dx, Hx, Wx) == (By, Dy, Hy, Wy)
        x, y = x.reshape(Bx, Px, Dx, Hx, Wx), y.reshape(Bx, 1, Dx, Hx, Wx)
        x = x * y
        return self.recon(x)

    def create_mask(self, x):
        """
        x: [B, P, D, H, W]
        """
        norm = torch.linalg.norm(x, dim=1, keepdim=True)
        return (norm > self.threshold).float()                          # return: [B, 1, D, H, W]


# ----------------------------------------------- Helper functions -----------------------------------------------

def dynamic_routing(votes, biases, routings):
    """
    Inputs:
        - votes: [B, Ci, Co, Po, Do, Ho, Wo]
        - biases:[1, 1, Co, Po, 1, 1, 1]
        - routings: number of dynamic routing iterations
    """
    B, Ci, Co, Po, Do, Ho, Wo = votes.shape                                 # votes: [B, Ci, Co, Po, Do, Ho, Wo]
    device = votes.device

    bij = torch.zeros(B, Ci, Co, 1, Do, Ho, Wo).to(device)                  # bij: [B, Ci, Co, 1, Do, Ho, Wo]

    for t in range(routings):
        cij = softmax(bij, dim=2)                                         # cij: [B, Ci, Co, 1, Do, Ho, Wo]
        sj = torch.sum(cij * votes, dim=1, keepdim=True) + biases           # sj: [B, 1, Co, Po, Do, Ho, Wo]
        vj = squash(sj)                                                     # vj: [B, 1, Co, Po, Do, Ho, Wo]
        if t < routings - 1:
            bij = bij + torch.sum(votes * vj, dim=3, keepdim=True)          # bij: [B, Ci, Co, 1, Do, Ho, Wo]

    return vj.squeeze(1)                                                    # return: [B, Co, Po, Do, Ho, Wo]



def squash(sj):
    """
    Inputs:
        - sj: [B, 1, Co, Po, Do, Ho, Wo]
    Output:
        - vj: [B, 1, Co, Po, Do, Ho, Wo]
    """
    sjnorm = torch.linalg.norm(sj, dim=3, keepdim=True)                     # sjnorm: [B, 1, Co, 1, Do, Ho, Wo]
    sjnorm2 = sjnorm ** 2
    return sjnorm2 * sj / ((1 + sjnorm2) * sjnorm)                          # return vj: [B, 1, Co, Po, Do, Ho, Wo]



def vector_norm(x):
    """
    Input:
        - x: [B, 1, P, D, H, W]
             x should have 1 capsule channel.
    Output:
        - norms of x: [B, 1, D, H, W]
    """
    assert x.shape[1] == 1
    return torch.linalg.norm(x, dim=2)      # return: [B, 1, D, H, W]



# -------------------------------------------------- CODE TESTING -----------------------------------------------------

if __name__ == '__main__':

    labels_list = [0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41, 43,
                   44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 77, 80, 85, 251, 252, 253, 254, 255, 1000]
    model = UNet3D(Ci=1, Co=len(labels_list))

    x = torch.rand((2, 1, 16, 16, 16))
    y = model(x)
    print(y.shape)
    print(y.sum(dim=1))


