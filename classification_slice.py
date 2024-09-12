import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import pandas as pd
import os
import cv2
import nibabel as nib
from torch.nn import Linear, LeakyReLU, BatchNorm1d, Dropout
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import wandb

# .....................................................................................................................
# Model Definitions
# .....................................................................................................................
def double_convolution(in_channels, out_channels):
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
        self.up_transpose_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_convolution_4 = double_convolution(128, 64)
        self.s_dp = nn.Dropout2d(p=0.5)

        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_4 = self.s_dp(down_4)
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
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = self._full_connect_set(1024, 512, dropout_p=0.5)
        self.fc2 = self._full_connect_set(512, 256, dropout_p=0.5)
        self.fc3 = Linear(256, 1)

    def forward(self, x):
        down_features = self.down_layers(x)
        pooled_features = self.global_avg_pool(down_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        x = self.fc1(pooled_features)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def _full_connect_set(self, in_features, out_features, dropout_p=0.5):
        return nn.Sequential(
            Linear(in_features, out_features),
            BatchNorm1d(out_features),
            LeakyReLU(negative_slope=0.02, inplace=True),
            Dropout(p=dropout_p),
        )

def freeze_layers(model):
    for param in model.down_layers.parameters():
        param.requires_grad = False

# .....................................................................................................................
# Loading Clinical Data
# .....................................................................................................................
df = pd.read_csv("/home/joshua_zhu/PCNSL/T1/data/csvs/pcnsl_clinical_data.csv")
class_counts = df['Ground Truth Binary Two Year'].value_counts()
minority_class_weight = class_counts.max() / class_counts.min()

if class_counts[0] > class_counts[1]:
    weights = {0: 1, 1: class_counts[0] / class_counts[1]}
else:
    weights = {0: class_counts[1] / class_counts[0], 1: 1}

# weights = {0: class_counts[1] / class_counts[0], 1: 1}

sample_weights = df['Ground Truth Binary Two Year'].map(weights)
sample_weights_tensor = torch.tensor(sample_weights.values, dtype=torch.float)
sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)

# .....................................................................................................................
# Loading Images
# .....................................................................................................................
def apply_transform(image):
    image = v2.RandomRotation(15)(image)
    image = v2.RandomHorizontalFlip(p=0.5)(image)
    image = v2.GaussianBlur(kernel_size=(3, 3), sigma=(.1, .3))(image)
    return image

def is_blank_slice(image):
    return np.max(image) == 0

def load_dataset(csv_path, root_dir, img_dim=(256, 256), test_size=0.2, augment=False):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1).reset_index(drop=True)
    images, labels, groups = [], [], []

    if augment:
        data_transform = transforms.Compose([
            transforms.Lambda(lambda x: apply_transform(x)),
        ])
    else:
        data_transform = None

    for _, row in df.iterrows():
        patient_id = row['Visage MRN']
        label = row['Ground Truth Binary Two Year']
        image_path = os.path.join(root_dir, f"{patient_id}/img_masked.nii.gz")

        if not os.path.exists(image_path):
            continue

        nifti_img = nib.load(image_path)
        img_data = nifti_img.get_fdata()

        for s in range(img_data.shape[2]):
            image = img_data[:, :, s]
            if is_blank_slice(image):
                continue

            image = image / np.max(image)
            image_resized = cv2.resize(image, img_dim, interpolation=cv2.INTER_AREA)

            if data_transform:
                transformed_data = data_transform(image_resized)
                image_resized = transformed_data

            images.append(image_resized.reshape(1, *img_dim))
            labels.append(label)
            groups.append(patient_id)

    images = np.stack(images)
    labels = np.array(labels)
    groups = np.array(groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=47)
    train_idx, test_idx = next(gss.split(images, labels, groups))

    X_train, X_test = images[train_idx], images[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

    return train_data, test_data

image_data_dir = '/home/joshua_zhu/PCNSL/T1/T1images'
train_data, test_data = load_dataset("/home/joshua_zhu/PCNSL/T1/data/csvs/pcnsl_clinical_data.csv", image_data_dir, augment=True)

print("Train set:", len(train_data))
print("Test set:", len(test_data))
print("Training set class distribution:", np.bincount(train_data.tensors[1].int()))
print("Testing set class distribution:", np.bincount(test_data.tensors[1].int()))

# .....................................................................................................................
# Model
# .....................................................................................................................
wandb.init(
    project="PCNSL-classification-binaryOS",
    config={
        "learning_rate": 0.001,
        "architecture": "LifeNet",
        "dataset": "Yale-PCNSL",
        "epochs": 100,
    }
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_weights(model, model_path):
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model

pretrained_model = UNet(num_classes=1)
pretrained_model_path = '/home/joshua_zhu/PCNSL/T1/T1images/best_finetunedmodel.pth'
load_model_weights(pretrained_model, pretrained_model_path)
pretrained_model.to(device)

model = LifeNet(pretrained_model)
freeze_layers(model)
model.to(device)

# .....................................................................................................................
# Training Loop
# .....................................................................................................................
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=10, min_delta=0.01)

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

pos_weight = torch.tensor([minority_class_weight], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

num_epochs = 200
best_val_auc = 0.0
best_epoch = 0
early_stopping = EarlyStopping(patience=10, min_delta=0.01)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_predictions, all_labels, all_probabilities = [], [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
        predictions = (probabilities > 0.5).astype(np.float32)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)

    accuracy = accuracy_score(all_labels, all_predictions)
    sensitivity = recall_score(all_labels, all_predictions, zero_division=0)
    specificity = recall_score(all_labels, all_predictions, pos_label=0, zero_division=0)
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, np.array(all_probabilities)) if len(np.unique(all_labels)) > 1 else 0

    print(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}, '
          f'Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}, '
          f'F1 Score: {f1}, AUC: {auc}')

    model.eval()
    val_running_loss = 0.0
    val_all_predictions, val_all_labels, val_all_probabilities = [], [], []

    with torch.no_grad():
        for val_inputs, val_labels in test_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels.unsqueeze(1))
            val_running_loss += val_loss.item()

            val_probabilities = torch.sigmoid(val_outputs).cpu().numpy()
            val_predictions = (val_probabilities > 0.5).astype(np.float32)

            val_all_labels.extend(val_labels.cpu().numpy())
            val_all_predictions.extend(val_predictions)
            val_all_probabilities.extend(val_probabilities)

    val_accuracy = accuracy_score(val_all_labels, val_all_predictions)
    val_sensitivity = recall_score(val_all_labels, val_all_predictions)
    val_specificity = recall_score(val_all_labels, val_all_predictions, pos_label=0)
    val_f1 = f1_score(val_all_labels, val_all_predictions)
    val_auc = roc_auc_score(val_all_labels, np.array(val_all_probabilities)) if len(
        np.unique(val_all_labels)) > 1 else 0

    wandb.log({
        "Epoch": epoch + 1,
        "Training Loss": running_loss / len(train_loader),
        "Training Accuracy": accuracy,
        "Training Sensitivity": sensitivity,
        "Training Specificity": specificity,
        "Training F1 Score": f1,
        "Training AUC": auc,
        "Validation Loss": val_running_loss / len(test_loader),
        "Validation Accuracy": val_accuracy,
        "Validation Sensitivity": val_sensitivity,
        "Validation Specificity": val_specificity,
        "Validation F1 Score": val_f1,
        "Validation AUC": val_auc
    })

    print(f'Epoch {epoch + 1}, Validation Loss: {val_running_loss / len(test_loader)}, '
          f'Accuracy: {val_accuracy}, Sensitivity: {val_sensitivity}, Specificity: {val_specificity}, '
          f'F1 Score: {val_f1}, AUC: {val_auc}')

    if val_auc > best_val_auc:
        print(f"New best model found at epoch {epoch + 1} with AUC {val_auc:.4f}. Saving model...")
        best_val_auc = val_auc
        best_epoch = epoch
        torch.save(model.state_dict(), '/home/joshua_zhu/PCNSL/T1/T1images/best_twoyearOSmodel_June3.pth')

    scheduler.step(val_running_loss / len(test_loader))

    early_stopping(val_running_loss / len(test_loader))
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break
