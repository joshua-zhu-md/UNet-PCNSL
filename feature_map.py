import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
import numpy as np
import pandas as pd
import os
import cv2
import nibabel as nib
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import BatchNorm1d
from torch.nn import Dropout
import torch.optim as optim
from torchvision import transforms as v2
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision import transforms
from sklearn.model_selection import train_test_split
import wandb
from torchvision.models.feature_extraction import create_feature_extractor


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

        diff_height = smaller_tensor.size(2) - larger_tensor.size(2)
        diff_width = smaller_tensor.size(3) - larger_tensor.size(3)

        # Calculate padding
        padding = [diff_width // 2, diff_width - diff_width // 2, diff_height // 2, diff_height - diff_height // 2]

        # Apply padding
        larger_tensor = F.pad(larger_tensor, padding)

        return larger_tensor

class LifeNet(nn.Module):
    def __init__(self, trained_model):
        super(LifeNet, self).__init__()
        self.down_layers = nn.Sequential(
            trained_model.down_convolution_1,
            trained_model.max_pool2d,
            trained_model.down_convolution_2,
            trained_model.max_pool2d,
            trained_model.down_convolution_3,
            trained_model.max_pool2d,
            trained_model.down_convolution_4,
            trained_model.max_pool2d,
            trained_model.down_convolution_5,
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # The output of global average pooling will be [batch_size, 1024, 1, 1], so the feature vector length is 1024
        self.fc1 = self._full_connect_set(1024, 512)
        self.fc2 = self._full_connect_set(512, 256)
        self.fc3 = Linear(256, 1)

        # self.fc1 = self._full_connect_set(1024, 512)
        # self.fc2 = Linear(512, 1)
        # self.act = Sigmoid()

    def forward(self, x):
        down_features = self.down_layers(x)
        pooled_features = self.global_avg_pool(down_features)

        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        x = self.fc1(pooled_features)
        x = self.fc2(x)
        x = self.fc3(x)
        output = x
        # output = self.act(x)
        return output

    def _full_connect_set(self, in_features, out_features):
        return nn.Sequential(
            Linear(in_features, out_features),
            BatchNorm1d(out_features),
            LeakyReLU(negative_slope=0.02, inplace=True),
            Dropout(p=0.5),
        )

def freeze_layers(model):
    # Freeze all layers in down_layers (UNet encoding layers)
    for param in model.down_layers.parameters():
        param.requires_grad = False

# .....................................................................................................................
# Loading Model
# .....................................................................................................................
def load_model_weights(model, model_path):
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model = UNet(num_classes=1)
pretrained_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_finetunedmodel.pth'
pretrained_model = load_model_weights(pretrained_model, pretrained_model_path)
pretrained_model.to(device)

model = LifeNet(pretrained_model)
model.to(device)

freeze_layers(model)

# OS_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_medianOSmodel.pth'
# OS_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_medianOSmodel_noaugments.pth'
# OS_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_oneyearOSmodel_noaugments.pth'
# OS_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_twoyearOSmodel_noaugments.pth'

OS_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_oneyearOSmodel_June3.pth'

model = load_model_weights(model, OS_model_path)
model.to(device)
model.eval()

return_nodes = {
    'fc2': 'out',
}

feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

# .....................................................................................................................
# Loading Images
# .....................................................................................................................
def apply_augmentation(image):
    image_pil = Image.fromarray(np.uint8(image * 255))

    transform = v2.Compose([
        v2.RandomRotation(15),
        v2.RandomHorizontalFlip(p=0.5),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.3))
    ])

    # Apply transformations
    image_augmented = transform(image_pil)

    # Convert back to numpy array and scale to [0, 1]
    image_augmented = np.array(image_augmented) / 255.0

    return image_augmented


def is_blank_slice(image):
    return np.max(image) == 0

def load_full_dataset(csv_path, root_dir, img_dim=(256, 256), batch_size=64):
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    patient_ids = []
    tumor_percentages = []

    for _, row in df.iterrows():
        patient_id = row['Visage MRN']
        label = row['Ground Truth Binary One Year']
        image_path = os.path.join(root_dir, f"{patient_id}/img_masked.nii.gz")

        if not os.path.exists(image_path):
            continue

        nifti_img = nib.load(image_path)
        img_data = nifti_img.get_fdata()
        total_tumor_volume = np.sum(img_data > 0)

        for s in range(img_data.shape[2]):
            image = img_data[:, :, s]

            if is_blank_slice(image):
                continue

            slice_tumor_volume = np.sum(image > 0)
            tumor_percentage = slice_tumor_volume / total_tumor_volume if total_tumor_volume > 0 else 0
            tumor_percentages.append(tumor_percentage)

            image = image / np.max(image)  # Normalize
            image_resized = cv2.resize(image, img_dim, interpolation=cv2.INTER_AREA)
            images.append(image_resized.reshape(1, *img_dim))
            labels.append(label)
            patient_ids.append(patient_id)

    images = np.stack(images)
    labels = np.array(labels).astype(float)
    tumor_percentages = np.array(tumor_percentages).astype(float)

    tensor_x = torch.tensor(images, dtype=torch.float32)
    tensor_y = torch.tensor(labels, dtype=torch.float32)
    tensor_tp = torch.tensor(tumor_percentages, dtype=torch.float32)

    dataset = TensorDataset(tensor_x, tensor_y, tensor_tp)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader, patient_ids

image_data_dir = '/home/joshua_zhu/PCNSL/T1/T1images'
csv_path = '/home/joshua_zhu/PCNSL/T1/data/csvs/pcnsl_clinical_data.csv'

# output_dir = '/home/joshua_zhu/PCNSL/T1glioma/features'
# output_dir = '/home/joshua_zhu/PCNSL/T1glioma/features2'
output_dir = '/home/joshua_zhu/PCNSL/T1glioma/features_greatestslice'

data_loader, patient_ids = load_full_dataset(csv_path, image_data_dir)

print (patient_ids)
# .....................................................................................................................
# Extract Features (All Slices)
# .....................................................................................................................
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# patient_slice_counter = {}
# patient_ids_iter = iter(patient_ids)
#
# for images, labels in data_loader:
#     images = images.to(device)
#     with torch.no_grad():
#         features = feature_extractor(images)['out'].cpu().numpy()
#
#     # Iterate over the batch and save features with patient ID and slice index
#     for i, image in enumerate(images):
#         patient_id = next(patient_ids_iter)
#
#         if patient_id not in patient_slice_counter:
#             patient_slice_counter[patient_id] = 0
#         else:
#             patient_slice_counter[patient_id] += 1
#
#         feature_path = os.path.join(output_dir, f"{patient_id}_{patient_slice_counter[patient_id]}.npy")
#         np.save(feature_path, features[i])
#
#     print(f"Processed {len(images)} images.")
#
# print("Feature extraction and saving complete.")

# .....................................................................................................................
# Extract Features (Weighted Slices)
# .....................................................................................................................
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# patient_slice_counter = {}
# patient_features = {}
# patient_ids_iter = iter(patient_ids)
#
# for images, labels, tumor_percentages in data_loader:
#     images = images.to(device)
#     with torch.no_grad():
#         features = feature_extractor(images)['out'].cpu().numpy()
#
#     for i in range(len(images)):
#         patient_id = next(patient_ids_iter)
#         tumor_percentage = tumor_percentages[i].item()
#
#         if patient_id not in patient_features:
#             patient_features[patient_id] = features[i] * tumor_percentage
#             patient_slice_counter[patient_id] = tumor_percentage
#         else:
#             patient_features[patient_id] += features[i] * tumor_percentage
#             patient_slice_counter[patient_id] += tumor_percentage
#
# # Verify that the weights for each patient sum up to 1
# for patient_id, total_tumor_percentage in patient_slice_counter.items():
#     print(f"Patient ID: {patient_id}, Total Tumor Percentage: {total_tumor_percentage}")
#
# for patient_id, features in patient_features.items():
#     total_tumor_percentage = patient_slice_counter[patient_id]
#     aggregated_features = features / total_tumor_percentage
#     feature_path = os.path.join(output_dir, f"{patient_id}.npy")
#     np.save(feature_path, aggregated_features)
#
# print("Feature extraction and saving complete.")

# .....................................................................................................................
# Extract Features (Greatest Weight Slice)
# .....................................................................................................................
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

patient_max_slice = {}
patient_ids_iter = iter(patient_ids)

for images, labels, tumor_percentages in data_loader:
    images = images.to(device)
    with torch.no_grad():
        features = feature_extractor(images)['out'].cpu().numpy()

    for i in range(len(images)):
        patient_id = next(patient_ids_iter)
        tumor_percentage = tumor_percentages[i].item()

        if patient_id not in patient_max_slice:
            patient_max_slice[patient_id] = (features[i], tumor_percentage)
        else:
            _, max_tumor_percentage = patient_max_slice[patient_id]
            if tumor_percentage > max_tumor_percentage:
                patient_max_slice[patient_id] = (features[i], tumor_percentage)

# Save features for the slice with the greatest weight for each patient
for patient_id, (features, tumor_percentage) in patient_max_slice.items():
    feature_path = os.path.join(output_dir, f"{patient_id}.npy")
    np.save(feature_path, features)

print("Feature extraction and saving complete.")
