"""
SelectorMLP: A PyTorch-based feature selection and classification framework.
"""

from .models import SelectorMLP, MultiClassifier
from .data import createDataset, createSVMDataset, evenClusters, chooseFeatures
from .split_data import splitData
from .feature_selection import selectFeaturesSelectorMLP, testFeatureSelection
from .visualization import plotFeatureSelectionResults, plotTrainingResults

__version__ = "2025.05.01" 