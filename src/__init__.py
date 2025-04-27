"""
SelectorMLP: A PyTorch-based feature selection and classification framework.
"""

from .models import SelectorMLP, MultiClassifier
from .data import createDataset, createSVMDataset, evenClusters, splitData, chooseFeatures
from .feature_selection import selectFeaturesSelectorMLP, testFeatureSelection
from .visualization import plotFeatureSelectionResults, plotTrainingResults

__version__ = "0.1.0" 