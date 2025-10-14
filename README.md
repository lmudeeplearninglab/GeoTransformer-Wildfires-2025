# GeoTransformer-Wildfires-2025

This project provides explainable AI tooling for wildfire modeling, originating from the `xai_for_wildfire` work. The original paper can be found [here](https://arxiv.org/pdf/2503.14150).

## Notebooks

**training_models.ipynb** is used for training all four models used in the paper, including baseline autoencoder, resnet, unet, and vit.

**XAI_SHAP_XGB.ipynb** and **XAI_SHAP_DL_Models.ipynb** are used for generating shap value for xgboost, baseline autoencoder, resnet, unet, and vit.

**XAI_GradCAM.ipynb** is used for generating gradcam heatmaps for baseline autoencoder, resnet, unet, and vit.

**XAI_Integradited_Gradients.ipynb** is used for generating ig feature contribution for baseline autoencoder, resnet, unet, and vit.

## Python utilities

The repository includes supporting utilities for training models and applying XAI techniques.
