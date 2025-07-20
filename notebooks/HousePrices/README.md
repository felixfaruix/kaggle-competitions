# House Prices - Advanced Regression Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-purple.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-red.svg)](https://numpy.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-yellow.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-lightgreen.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.0+-blue.svg)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-3.0+-lightblue.svg)](https://optuna.org/)

## Project Overview

An advanced machine learning pipeline for predicting house prices using ensemble learning, automated hyperparameter optimization, and feature engineering. This project involves production-ready ML practices including custom sklearn transformers, statistical outlier detection, and automated model selection.

### **Custom Transformers**
- `LogTransformer`: Skewness reduction with sparse feature detection
- `OutliersRemoval`: Statistical outlier elimination for training data
- `CatEncoder`: Combined one-hot and ordinal encoding with missing value handling
- `TotalArea`: Domain-specific feature combination (basement + ground floor)
- `TotalBaths`: Weighted bathroom counting (full=1.0, half=0.5)
- `HighlyCorrelatedFeatures`: Automated multicollinearity reduction
- `MedianImputer`: Robust missing value imputation
- `AgeCalculator`: Temporal feature engineering (house age from build/sale years)

### **Model Evaluation Pipeline**
- **Optuna Objective Function**: Automated ensemble discovery with cross-validation scoring
- **Performance Metrics**: RMSE, R², MAE for comprehensive evaluation
- **Residual Analysis**: Error distribution examination for model validation
- **Visualization**: Optimization history and parameter importance plots

### **Data Processing Strategy**
- **Training Pipeline**: Includes outlier removal and feature engineering
- **Validation/Test Pipeline**: Consistent preprocessing without outlier removal
- **Missing Value Handling**: Differential imputation

#### **Getting Started**:

```bash
pip install -r requirements.txt
```