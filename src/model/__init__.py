# -*- coding: utf-8 -*-
"""
Model Module
============

This module contains the DeepText Multi-Task Learning model implementations.

Classes:
- DeepTextMultiTaskClassifier: Basic multi-task classifier
- DeepTextMultiTaskClassifierOptimized: Optimized version with attention and batch normalization
"""

from .deeptext_multitask import DeepTextMultiTaskClassifier
from .multi_task_model_optimized import DeepTextMultiTaskClassifierOptimized

__all__ = ['DeepTextMultiTaskClassifier', 'DeepTextMultiTaskClassifierOptimized']














