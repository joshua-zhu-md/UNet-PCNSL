"""
nnCapsNet Project
Developed by Arman Avesta, MD
Aneja Lab | Yale School of Medicine
Created (11/1/2022)
Updated (11/1/2022)

This file contains functions and classes to compute different types of losses.
"""

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:


# System imports:
import numpy as np
import torch
from torch import nn

# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)

# ---------------------------------------------- MAIN CLASSES / FUNCTIONS ---------------------------------------------

class DiceLoss(nn.Module):
	def __init__(self, conversion='sigmoid'):
		"""
		Inputs:
			- conversion: see convert_preds function below to see the possible options.
		"""
		super().__init__()
		self.conversion = conversion
		self.sigmoid = nn.Sigmoid()

	def forward(self, preds, targets):
		"""
		Inputs:
			- preds: predictions = model outputs = proposed segmentations
			- targets: ground truth = actual segmentations
			preds and targets should have the same shape: [B, C, D, H, W]
														(or [B, C, H, W] for 2D models)
			B: batches, C: channels, D: depth, H: height, W: width.
		Output:
			- dice score: scalar if reduction='mean', and a tensor with B components if reduction='none'.
		"""
		e = 0.0001
		# preds = self.sigmoid(preds)
		preds = torch.sigmoid(preds)
		preds, targets = preds.flatten(), targets.flatten()
		intersection = (preds * targets).sum()
		dice_loss = 1 - (2 * intersection + e) / (preds.sum() + targets.sum() + e)
		return dice_loss

# .....................................................................................................................

class DiceBCELoss(nn.Module):
	def __init__(self, conversion='sigmoid'):
		"""
		Inputs:
			- conversion: see convert_preds function below to see the possible options.
		"""
		super().__init__()
		self.dice = DiceLoss(conversion=conversion)
		self.bcelogitloss = nn.BCEWithLogitsLoss(reduction='mean')
		self.bceloss = nn.BCELoss(reduction='mean')

	def forward(self, preds, targets):
		# Calculate Dice:
		dice_loss = self.dice(preds, targets)
		# Calculate BCE loss:
		bce_loss = self.bcelogitloss(preds, targets)
		# bce_loss = self.bceloss(preds, targets)
		print(f'dice loss: {dice_loss}, bce_loss: {bce_loss}')
		return (dice_loss + bce_loss) / 2


# ------------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------

def convert_preds(preds, conversion='sigmoid', low=0.1, high=0.9):
	"""
	predictions (model outputs) --> converted predictions (to be used in loss function)
	Inputs:
		- preds: predictions = model outputs = proposed segmentations.
		- conversion: how preds should be converted. Options: 'margin', 'threshold', 'sigmoid', 'logit', 'none'
		- low: low margin; only used if coversion is set to 'margin'.
		- high: high margin; only used of coversion is set to 'margin'.

	Output:
	- converted predictions
	Conversion options:
		- 'margin': preds --> 0 if preds < low; preds if low < preds < high; 1 if preds > high
		- 'threshold': preds --> 0 if preds < 0.5; 1 if preds >= 0.5
		- 'sigmoid': preds --> sigmoid(preds)
		- 'none': returnds preds themselves
	"""
	if conversion == 'sigmoid':
		return torch.sigmoid(preds)

	if conversion == 'margin':
		z = torch.zeros_like(preds)
		z[preds > low] = (preds[preds > low] - low) / (high - low)
		z[preds > high] = 1
		return z

	if conversion == 'threshold':
		return torch.round(preds)

	if conversion == 'none':
		return preds


# .....................................................................................................................




# -------------------------------------------------- CODE TESTING -----------------------------------------------------

if __name__ == '__main__':
	print(f'''

    ''')
