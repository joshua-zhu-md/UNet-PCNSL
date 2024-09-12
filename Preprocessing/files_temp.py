import os
import shutil


root_folder = "/home/joshua_zhu"
target_folder = "/home/joshua_zhu/PCNSL/T1glioma/T1gliomaimages"

# ---------------------------------------------- Moving files ---------------------------------------------------------
for patient in os.listdir(root_folder):

    patient_path = os.path.join(root_folder, patient)
    # checking if it is a file
    if os.path.isdir(patient_path) and ('YG_' in patient_path):
        os.chdir(patient_path)
        print (patient)

        shutil.move(patient_path, target_folder)