import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import cv2
from torchsummary import summary
from sklearn.utils import shuffle
import nibabel as nib
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Sigmoid
from torch.nn import Flatten
import torch.optim as optim
from torchvision import transforms as v2
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision import transforms
from sklearn.model_selection import train_test_split
import wandb

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

        padding = [diff_width // 2, diff_width - diff_width // 2, diff_height // 2, diff_height - diff_height // 2]

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
        self.fc3 = self._full_connect_set(256, 128)
        self.fc4 = Linear(128, 1)

    def forward(self, x):
        down_features = self.down_layers(x)
        pooled_features = self.global_avg_pool(down_features)

        # Since the global average pooling output is [batch_size, 1024, 1, 1], we need to remove the last two dimensions
        # before feeding it into the fully connected layers
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        x = self.fc1(pooled_features)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
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
OS_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_oneyearOSmodel_noaugments.pth'
# OS_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_twoyearOSmodel_noaugments.pth'

model = load_model_weights(model, OS_model_path)
model.to(device)

# .....................................................................................................................
# Loading Images
# .....................................................................................................................
# validation_patient_ids = np.load("two_year_validation_patient_ids.npy")
# validation_patient_ids = ['YG_0MIZJYTGCGOC', 'YG_1IAGMX7VXNH2', 'YG_25OPOBHYWB94', 'YG_25OPOBHYWB94', 'YG_2DC8H8FZMMAS',
#                           'YG_2JCL8FTYC4V4', 'YG_2NCN4IGSEM64', 'YG_4QPPT1P9OUAH', 'YG_5B4JU7KC832H', 'YG_BIP3344CD1JF',
#                           'YG_CP100KBM76WS', 'YG_CUAP4BJ0BDEX', 'YG_CWAW70G2SD1R', 'YG_CZZK5YEL3EPI', 'YG_E0J1XL84B33B',
#                           'YG_EB16BAB13NJ8', 'YG_EKR3V5WUU1WO', 'YG_EN2GZC31Q2EH', 'YG_GWKZMDDIMCYV', 'YG_H4NXE3UTNTA8',
#                           'YG_I7A0I2SIQLWS', 'YG_IOWV90WJ0VTJ', 'YG_NH04HY6C7I0X', 'YG_NZSSCYAYAHO9', 'YG_PQEAX3VLO20M',
#                           'YG_PUPUOD8H190W', 'YG_RDHVFX1R04BL', 'YG_SGGJL5RKXDDQ', 'YG_TA40BGW2XEN5', 'YG_TO0PRDBVEPIC',
#                           'YG_TXOAKBX8LGVV', 'YG_TXX10ILKHJ7Q', 'YG_TYQZX8NMLHQE', 'YG_VBB4NNF0IIS3', 'YG_YWCWIAXNCT66',
#                           'YG_Z5EE4IUVWBK1', 'YG_AIWT87YPOJ6J']

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
def load_patient_slices(csv_path, root_dir, img_dim=(256, 256), include_patients=None):
    df = pd.read_csv(csv_path)
    patients = []
    patient_ids = []
    patient_labels = []

    # If include_patients is specified, convert to set for faster lookups
    if include_patients is not None:
        include_patients = set(include_patients)

    for _, row in df.iterrows():
        patient_id = row['Visage MRN']
        if include_patients is not None and patient_id not in include_patients:
            continue  # Skip patients not in the include list

        patient_label = row['Ground Truth Binary One Year']
        image_path = os.path.join(root_dir, patient_id, 'img_masked.nii.gz')

        if not os.path.exists(image_path):
            print(f"Error: The file {image_path} does not exist or is not accessible.")
            continue

        nifti_img = nib.load(image_path)
        img_data = nifti_img.get_fdata()

        patient_slices = []
        for s in range(img_data.shape[2]):
            image = img_data[:, :, s]
            if np.max(image) == 0:
                continue

            image = image / np.max(image)
            image_resized = cv2.resize(image, img_dim, interpolation=cv2.INTER_AREA)
            patient_slices.append(image_resized)

        if patient_slices:
            patient_slices = np.array(patient_slices).reshape(-1, 1, img_dim[0], img_dim[1])
            patients.append(patient_slices)
            patient_ids.append(patient_id)
            patient_labels.append(patient_label)

    return patients, patient_ids, patient_labels


csv_path = "/home/joshua_zhu/PCNSL/T1/data/csvs/pcnsl_clinical_data.csv"
image_data_dir = '/home/joshua_zhu/PCNSL/T1/T1images'

# Adjust the call to load_patient_slices to only include validation patients
patients, patient_ids, patient_labels = load_patient_slices(csv_path, image_data_dir)

# print(f"Total patients loaded: {len(patients)}")

# .....................................................................................................................
# Patient Predictions
# .....................................................................................................................
# Assume patient_labels are correctly loaded corresponding to each patient
patient_data = [torch.tensor(patient, dtype=torch.float) for patient in patients]

predicted_labels = []  # For patient-level prediction
true_labels = []  # For patient-level true labels

predicted_slices = []  # For slice-level prediction
true_slices = []  # For slice-level true labels

def predict_patient(patient_slices, model, patient_true_label):
    patient_votes = []

    for slice_data in patient_slices:
        slice_data = slice_data.unsqueeze(0).to(device)  # Add batch dimension and move to device
        with torch.no_grad():
            model.eval()
            outputs = model(slice_data)
            probability = torch.sigmoid(outputs).cpu().numpy()
            vote = 1 if probability > 0.5 else 0
            patient_votes.append(vote)

            # Append predictions for each slice and the corresponding true label
            predicted_slices.append(vote)
            true_slices.append(patient_true_label)  # Assuming the label is the same for all slices of a patient

    # Count the votes for "Yes" (1) and "No" (0)
    num_yes_votes = sum(patient_votes)
    num_no_votes = len(patient_votes) - num_yes_votes

    total_votes = num_yes_votes + num_no_votes

    # Majority voting decision
    # final_prediction = 1 if num_yes_votes > num_no_votes else 0
    final_prediction = 1 if num_yes_votes > (total_votes * 0.3) else 0

    return final_prediction

# Iterate over each patient
for patient_slices, patient_id, label in zip(patient_data, patient_ids, patient_labels):
    final_prediction = predict_patient(patient_slices, model, label)

    predicted_labels.append(final_prediction)
    true_labels.append(label)

    print(f'Patient {patient_id}: Prediction - {final_prediction}, True Label - {label}')

# Calculate metrics for patient-level predictions
accuracy_patients = accuracy_score(true_labels, predicted_labels)
sensitivity_patients = recall_score(true_labels, predicted_labels)  # True Positive Rate
specificity_patients = recall_score(true_labels, predicted_labels, pos_label=0)  # True Negative Rate
precision_patients = precision_score(true_labels, predicted_labels)

print(
    f"Patient-level Accuracy: {accuracy_patients:.4f}, Sensitivity: {sensitivity_patients:.4f}, Specificity: {specificity_patients:.4f}, Precision: {precision_patients:.4f}"
)

# Calculate metrics for slice-level predictions
accuracy_slices = accuracy_score(true_slices, predicted_slices)
print(f"Slice-level Accuracy: {accuracy_slices:.4f}")
