# UNet PCNSL

This repository contains the code for our Unet-based deep learning model which predicts overall survival (OS) for primary central nervous system (PCNSL).

We have described the dataset, our models, and our results in the paper "Deriving Imaging Biomarkers for Primary Central Nervous System Lymphoma Using Deep Learning". 
The pre-print can be accessed at: https://doi.org/10.1101/2022.01.18.22269482.


Files description:

pre_processing: preprocesses the patient MRI as described in our paper

pretraining: pretrain Unet on glioma dataset

finetuning: transfer learning approach on main PCNSL dataset

classification_slice: main model for slice-level prediction

classification_patient: implementation of voting system for patient-level prediction

subgroups: split dataset in subanalysis groups as described in our paper

feature_map: extraction of quantitiative biomarkers for interpretability 


Author:
Joshua Zhu
