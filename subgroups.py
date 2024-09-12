import os
import nibabel as nib
import numpy as np
import pandas as pd
import cv2

csv_path = "/home/joshua_zhu/PCNSL/T1/data/csvs/pcnsl_clinical_data.csv"
root_dir = "/home/joshua_zhu/PCNSL/T1/T1images"

all_folders = []
for d in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('YG_'):
        all_folders.append(d)

def load_patient_slices_and_masks(csv_path, root_dir, img_dim=(256, 256)):
    df = pd.read_csv(csv_path)
    patient_data = []

    for _, row in df.iterrows():
        patient_id = row['Visage MRN']
        label = row['Binary Median']
        image_path = os.path.join(root_dir, patient_id, 'img_preprocessed.nii.gz')
        mask_path = os.path.join(root_dir, patient_id, 'seg_preprocessed.nii.gz')

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(patient_id)
            continue

        nifti_img = nib.load(image_path)
        nifti_mask = nib.load(mask_path)
        img_data = nifti_img.get_fdata()
        mask_data = nifti_mask.get_fdata()

        total_tumor_volume = np.sum(mask_data > 0)  # Sum over the entire volume

        patient_data.append({
            'patient_id': patient_id,
            'label': label,
            'total_tumor_volume': total_tumor_volume  # Changed from slices to total volume
        })

    return patient_data

def subgroup_analysis(patient_data):
    # Convert total tumor volumes to a list for global statistics
    all_volumes = [p['total_tumor_volume'] for p in patient_data]

    # Define thresholds based on total tumor volume
    small_threshold = np.percentile(all_volumes, 50)
    large_threshold = np.percentile(all_volumes, 50)

    print (small_threshold)
    print (large_threshold)

    # Function to classify and count slices based on volume
    def classify_and_count(volume):
        return volume <= small_threshold, volume >= large_threshold

    small_tumor_patients = []
    large_tumor_patients = []
    multifocal_tumor_slices = 0

    # Patient IDs for multifocal tumors
    multifocal_patient_ids = ['YG_0MIZJYTGCGOC', 'YG_1IAGMX7VXNH2', 'YG_25OPOBHYWB94', 'YG_25OPOBHYWB94', 'YG_2DC8H8FZMMAS',
                              'YG_2JCL8FTYC4V4', 'YG_2NCN4IGSEM64', 'YG_4QPPT1P9OUAH', 'YG_5B4JU7KC832H', 'YG_BIP3344CD1JF',
                              'YG_CP100KBM76WS', 'YG_CUAP4BJ0BDEX', 'YG_CWAW70G2SD1R', 'YG_CZZK5YEL3EPI', 'YG_E0J1XL84B33B',
                              'YG_EB16BAB13NJ8', 'YG_EKR3V5WUU1WO', 'YG_EN2GZC31Q2EH', 'YG_GWKZMDDIMCYV', 'YG_H4NXE3UTNTA8',
                              'YG_I7A0I2SIQLWS', 'YG_IOWV90WJ0VTJ', 'YG_NH04HY6C7I0X', 'YG_NZSSCYAYAHO9', 'YG_PQEAX3VLO20M',
                              'YG_PUPUOD8H190W', 'YG_RDHVFX1R04BL', 'YG_SGGJL5RKXDDQ', 'YG_TA40BGW2XEN5', 'YG_TO0PRDBVEPIC',
                              'YG_TXOAKBX8LGVV', 'YG_TXX10ILKHJ7Q', 'YG_TYQZX8NMLHQE', 'YG_VBB4NNF0IIS3', 'YG_YWCWIAXNCT66',
                              'YG_Z5EE4IUVWBK1', 'YG_AIWT87YPOJ6J']

    for patient in patient_data:
        is_small, is_large = classify_and_count(patient['total_tumor_volume'])
        if is_small:
            small_tumor_patients.append(patient['patient_id'])
        if is_large:
            large_tumor_patients.append(patient['patient_id'])

        # Print the results
    print("Small tumor patients:", len(small_tumor_patients), small_tumor_patients)
    print("Large tumor patients:", len(large_tumor_patients), large_tumor_patients)
    print(f"Multifocal tumor slices: {multifocal_tumor_slices}")

    non_counted_folders = [folder for folder in all_folders if
                           folder not in small_tumor_patients and folder not in large_tumor_patients]

    print(non_counted_folders)

patient_data = load_patient_slices_and_masks(csv_path, root_dir)
subgroup_analysis(patient_data)

# print(sum(1 for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('YG_')))
