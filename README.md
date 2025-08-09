# Wine Classification — SHAP Feature Importance Analysis

## Overview
This project compares feature importance across multiple machine learning models for a wine classification task, using **SHAP (SHapley Additive exPlanations)**.  
The aim is to understand **which wine properties most influence predictions** and whether different algorithm types agree on the same important features.

By analyzing SHAP values, we can see not only **how much** each feature matters but also **how** it impacts predictions — making the models more interpretable and trustworthy.

---

## Dataset
The dataset contains measurements of various **chemical** and **visual** properties of wines, such as:
- **Proline**
- **Alcohol**
- **Flavanoids**
- **Color Intensity**
- **Hue**
- **OD280/OD315 of diluted wines**
- **Malic Acid**, **Alcalinity of Ash**, and others.

Each sample belongs to one of several wine classes.

---

## Models Analyzed
Eight models from different machine learning families were trained:

- **Tree-based:** Decision Tree (CART), Random Forest, Gradient Boosting  
- **Probabilistic:** Gaussian Naive Bayes  
- **Linear:** SVM (Linear)  
- **Kernel-based:** SVM (RBF)  
- **Instance-based:** k-Nearest Neighbors  
- **Neural Network:** Multi-layer Perceptron (MLP)  

This diversity lets us see whether feature importance is consistent across fundamentally different learning strategies.

---

## SHAP Analysis Process
1. **Training:** Each model was trained on the wine dataset.
2. **Explanation:** SHAP values were calculated for the same test set.
3. **Saving results:** For each model, the script outputs:
   - **Global Importance Bar Charts** — average absolute SHAP values per feature.
   - **Beeswarm Plots** — distribution of SHAP values per sample.
   - **CSV files** with numeric SHAP importances.
   - **Raw SHAP arrays** for advanced analysis.

---

## Model Family Interpretability

### 1. Tree-based Models  
**Includes:** Decision Tree (CART), Random Forest, Gradient Boosting  
**Frequent top features:** `color_intensity`, `flavanoids`, `proline`

**Why similar:**  
Tree algorithms split data into smaller, more homogeneous groups.  
At each split, they choose the **feature and threshold** that best separates classes — creating the **sharpest partition**.  
In this dataset, `color_intensity` and `flavanoids` create especially clear class boundaries, so all three models consistently rank them highest.  
Even though the models differ in complexity (single tree vs. ensembles), their SHAP rankings align because they share this split-selection logic.

---

### 2. Probabilistic Model  
**Includes:** Gaussian Naive Bayes  
**Frequent top features:** `flavanoids`, `color_intensity`, `proline`, `alcohol`, `od280/od315_of_diluted_wines`

Gaussian Naive Bayes applies **Bayes’ theorem** under the assumption that features are **independent given the class**.  
Despite treating predictors separately, it still ranks `flavanoids`, `color_intensity`, and `proline` as its top three predictors.  
That a completely different algorithm type highlights the same features reinforces their importance in explaining wine classification.

---

## Key Findings
- **Universal top features:** `proline`, `flavanoids`, `color_intensity`, and `alcohol` appear across nearly all models.
- **Tree-based** models focus heavily on visual metrics (`color_intensity`).
- **SVMs and k-NN** emphasize chemical composition (`proline`, `alcohol`).
- **Naive Bayes** spreads importance evenly but still ranks the same top predictors highly.
- **Neural Net** blends chemical and color features without over-relying on one.

---

## Project Structure
## Project Structure
```plaintext
shap_exports/                  # SHAP outputs for each model
├─ <Model_Name>/               # Folder for a specific model
│  ├─ global_importance.png    # Global bar chart of mean |SHAP| values
│  ├─ beeswarm_class_<name>.png# Beeswarm plot for a specific class
│  ├─ mean_abs_shap_by_feature.csv # Numeric feature importances
│  ├─ raw_shap_values.npz      # Raw SHAP arrays for advanced analysis
│  └─ X_explain.csv            # Data points used in SHAP calculation
├─ manifest.json                # Metadata about all models' outputs
shap_exports.zip                # Zipped export for sharing
analysis_script.py              # SHAP computation and export script
README.md                       # This document

