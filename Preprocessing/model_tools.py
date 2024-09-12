"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This file contains tools to help save and load models.
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:
from os_tools import print_and_log


# System imports:
import numpy as np
import torch
import os
from os.path import join
from datetime import  datetime
from tqdm import tqdm

# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)

# ---------------------------------------------- MAIN CLASSES / FUNCTIONS ---------------------------------------------



# ------------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------

def save_model(model, model_path):
    checkpoint = {'state_dict': model.state_dict()}
    # checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
    torch.save(checkpoint, model_path)
    print(f'''
    >>>   MODEL SAVED   <<<
    ''')

# .....................................................................................................................

def load_model(model, saved_model_path):
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'>>>   Loaded the model from: {saved_model_path}   <<<')
    return model


# -------------------------------------------------- CODE TESTING -----------------------------------------------------

if __name__ == '__main__':
    print(f'''

    ''')
