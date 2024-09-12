import torch
import nibabel as nib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import wandb
from torchvision import transforms
from torchvision.transforms import v2

# .....................................................................................................................
TRAIN_DATASET_PATH = '/home/joshua_zhu/PCNSL/T1glioma/T1gliomaimages/all_data/'
# VALIDATION_DATASET_PATH = '/home/joshua_zhu/PCNSL/T1glioma/T1gliomaimages/val_data/'

torch.cuda.is_available()
# .....................................................................................................................

# .....................................................................................................................
# Hyperparameters
# .....................................................................................................................
batch_size = 16
torch.manual_seed(42)
epochs = 1000
learning_rate = 0.001

# .....................................................................................................................
# Model Definitions
# .....................................................................................................................

def double_convolution(in_channels, out_channels):
    # Convolution operations have padding because the output result size needs to be same as input size.
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
        nn.ReLU(inplace=True)
    )
    return conv_op

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down (contracting path)
        self.down_convolution_1 = double_convolution(1, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        # Up (expanding path)
        # Below, `in_channels` again becomes 1024 as we are concatenating
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2,
            stride=2)
        self.up_convolution_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2,
        )
        self.up_convolution_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_convolution_3 = double_convolution(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_convolution_4 = double_convolution(128, 64)

        self.s_dp = nn.Dropout2d(p=0.5)

        # Output is `out_channels` (as per the number of classes)
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)

        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)

        down_4= self.s_dp(down_4)

        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)

        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)

        down_8 = self.s_dp(down_8)

        down_9 = self.down_convolution_5(down_8)

        up_1 = self.up_transpose_1(down_9)
        up_2 = self.up_convolution_1(torch.cat([down_7, self.pad_to_match_size(up_1, down_7)], 1))

        up_2 = self.s_dp(up_2)

        up_3 = self.up_transpose_2(up_2)
        up_4 = self.up_convolution_2(torch.cat([down_5, self.pad_to_match_size(up_3, down_5)], 1))

        up_5 = self.up_transpose_3(up_4)
        up_6 = self.up_convolution_3(torch.cat([down_3, self.pad_to_match_size(up_5, down_3)], 1))

        up_6 = self.s_dp(up_6)

        up_7 = self.up_transpose_4(up_6)
        up_8 = self.up_convolution_4(torch.cat([down_1, self.pad_to_match_size(up_7, down_1)], 1))

        out = self.out(up_8)

        return out

    def pad_to_match_size(self, larger_tensor, smaller_tensor):
        """
        Pad the larger tensor to match the size of the smaller tensor along the spatial dimensions.
        """
        diff_height = smaller_tensor.size(2) - larger_tensor.size(2)
        diff_width = smaller_tensor.size(3) - larger_tensor.size(3)

        # Calculate padding
        padding = [diff_width // 2, diff_width - diff_width // 2, diff_height // 2, diff_height - diff_height // 2]

        # Apply padding
        larger_tensor = F.pad(larger_tensor, padding)

        return larger_tensor


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

# .....................................................................................................................
# Renaming patient directories
# .....................................................................................................................

train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
train_and_val_directories.remove(TRAIN_DATASET_PATH + 'images')

def pathListIntoIds(dirList):
    x = [dir_entry[dir_entry.rfind('/') + 1:] for dir_entry in dirList]
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories)

# Check if the dataset is large enough for the specified split
if len(train_and_test_ids) > 1:
    train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.1)
    train_ids, test_ids = train_test_split(train_test_ids, test_size=0.1)

    print(f"Train: {len(train_ids)} | Validation: {len(val_ids)} | Test: {len(test_ids)}")
else:
    print("Dataset is too small for the specified split.")

# .....................................................................................................................
# Data Loader
# .....................................................................................................................

# .....................................................................................................................
# Helper Functions
# .....................................................................................................................
print("Size of train_ids:", len(train_ids))

def load_dataset(ids, path, augment=True):
    print("Loading dataset...")

    images_list = []
    masks_list = []

    # Define data augmentation transforms
    if augment:
        data_transform = transforms.Compose([
            transforms.Lambda(lambda x: apply_transform(x)),
        ])
    else:
        data_transform = None

    for id in ids:
        if id.startswith("YG_"):
            t1 = nib.load(f"{path}images/{id}_img_preprocessed.nii.gz").get_fdata()
            seg = nib.load(f"{path}masks/{id}_seg_preprocessed.nii.gz").get_fdata()

            for s in range(4, seg.shape[2]-4, 10):
                image = t1[:, :, s] / t1.max()
                mask = seg[:, :, s] > 0

                if data_transform:
                    # Apply the custom transformation to both images and masks
                    transformed_data = data_transform((image, mask))
                    image, mask = transformed_data

                images_list.append(image)
                masks_list.append(mask)

    images = np.expand_dims(np.array(images_list), axis=1)
    masks = np.expand_dims(np.array(masks_list), axis=1)

    return images, masks


def apply_transform(data):
    image, mask = data
    image = v2.RandomRotation(15)(image)
    image = v2.RandomHorizontalFlip(p=0.5)(image)
    image = v2.GaussianBlur(kernel_size=(3, 3), sigma=(.1, .3))(image)
    mask = v2.RandomRotation(15)(mask)
    mask = v2.RandomHorizontalFlip(p=0.5)(mask)
    mask = v2.GaussianBlur(kernel_size=(3, 3), sigma=(.1, .3))(mask)

    return image, mask

# .....................................................................................................................
# Loading the images
# .....................................................................................................................

train_images, train_masks = load_dataset(train_ids, TRAIN_DATASET_PATH)
val_images, val_masks = load_dataset(val_ids, TRAIN_DATASET_PATH)

train_images.shape, val_masks.shape
print("Size of loaded dataset:", len(train_images))


# Training Images
train_dataset = TensorDataset(torch.from_numpy(train_images).type(torch.float32), torch.from_numpy(train_masks).type(torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Test (i.e. validation) Images
test_dataset = TensorDataset(torch.from_numpy(val_images).type(torch.float32), torch.from_numpy(val_masks).type(torch.float32))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# .....................................................................................................................
# Model Setup
# .....................................................................................................................
wandb.init(
    project="glioma-pretraining",

    config={
        "learning_rate": 0.001,
        "architecture": "UNet",
        "dataset": "Yale-Glioma",
        "epochs": 1000,
    }
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(num_classes=1).to(device)

# For additional epochs
# model.load_state_dict(torch.load('/home/joshua_zhu/PCNSL/T1glioma/T1gliomaimages/trainedmodel3.pth'))

wandb.watch(model, log_freq=10)

loss_fn = nn.BCEWithLogitsLoss()
loss_fn_1 = DiceLoss()
loss_fn_2 = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=0.001)

# additional_epochs = 5
# .....................................................................................................................
# Create training and testing loop
# .....................................................................................................................
best_val_loss = float('inf')

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch+1} of {epochs}")

# for epoch in tqdm(range(additional_epochs)):
#     print(f"Epoch: {epoch} of {additional_epochs}")
    ### Training
    train_loss_1, train_loss_2, train_loss = 0, 0, 0
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):

        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate loss (per batch)
        loss_1 = loss_fn_1(y_pred, y)
        loss_2 = loss_fn_2(y_pred, y)
        loss = loss_1 + loss_2
        train_loss += loss # add up the loss per epoch)
        train_loss_1 += loss_1
        train_loss_2 += loss_2

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer
        optimizer.step()

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    train_loss_1 /= len(train_dataloader)
    train_loss_2 /= len(train_dataloader)

    ### Testing
    test_loss_1, test_loss_2, test_loss = 0, 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in test_dataloader:

            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Calculate loss
            loss_1 = loss_fn_1(y_pred, y)
            loss_2 = loss_fn_2(y_pred, y)
            loss = loss_1 + loss_2
            test_loss += loss
            test_loss_1 += loss_1
            test_loss_2 += loss_2

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)
        test_loss_1 /= len(test_dataloader)
        test_loss_2 /= len(test_dataloader)

    ## Log outputs to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_DiceLoss": train_loss_1,
        "train_BCEWithLogitsLoss": train_loss_2,
        "test_loss": test_loss,
        "test_DiceLoss": test_loss_1,
        "test_BCEWithLogitsLoss": test_loss_2,
    })

    # Save the model if the current validation loss is better than the previous best
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        torch.save(model.state_dict(), '/home/joshua_zhu/PCNSL/T1glioma/T1gliomaimages/best_model.pth')

    # Log example images for visualization
    if epoch % 10 == 0:
        with torch.no_grad():
            example_images = []  # List to store example images
            for i in range(min(3, len(test_dataloader))):
                X, y = next(iter(test_dataloader))
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                example_images.append(wandb.Image(X[0].cpu(), caption="Input Image"))
                example_images.append(wandb.Image(y[0].cpu(), caption="Ground Truth Mask"))
                example_images.append(wandb.Image(torch.sigmoid(y_pred[0]).cpu(), caption="Predicted Mask"))

            wandb.log({"Examples": example_images})

    ## Print out what's happening
    print(f"Train loss: {train_loss:.5f}, Dice: {train_loss_1:.5f}, BCE: {train_loss_2:.5f} | Test loss: {test_loss:.5f}, Dice: {test_loss_1:.5f}, BCE: {test_loss_2:.5f}\n")

# torch.save(model.state_dict(), '/home/joshua_zhu/PCNSL/T1glioma/T1gliomaimages/trainedmodel5.pth')