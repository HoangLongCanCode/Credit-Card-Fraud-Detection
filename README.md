# Credit Card Fraud Detection

A comprehensive machine learning project comparing multiple tree-based algorithms for detecting fraudulent credit card transactions using highly imbalanced data.

## 📊 Project Overview

This project implements and compares four machine learning models to identify fraudulent credit card transactions:
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

The models are trained on the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, which contains transactions made by European cardholders in September 2013.

## 🎯 Key Features

- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Over-sampling Technique) to address the severe class imbalance (fraudulent transactions represent only 0.17% of all transactions)
- **Comprehensive Model Comparison**: Evaluates four different algorithms across multiple metrics
- **Feature Engineering**: Standardizes the `Amount` feature for better model performance
- **Model Persistence**: Saves all trained models and automatically selects the best performer

## 📈 Results

### Model Performance Comparison

| Model | AUC Score | Key Observations |
|-------|-----------|------------------|
| Decision Tree | 0.89 | Baseline model, prone to overfitting |
| Random Forest | 0.97 | Strong ensemble performance |
| XGBoost | 0.98 | Best overall performance |
| LightGBM | 0.96 | Fast training, competitive accuracy |

### Evaluation Metrics

Each model is evaluated using:
- **Confusion Matrix**: Shows true/false positives and negatives
- **ROC Curve**: Visualizes trade-off between true positive rate and false positive rate
- **Precision-Recall Curve**: Critical for imbalanced datasets, shows precision vs recall trade-off
- **Feature Importance**: Identifies which features contribute most to predictions

### Key Insights

From the visualization plots:

1. **XGBoost** demonstrates the highest AUC (0.98) with excellent precision-recall balance
2. **Random Forest** shows strong performance (AUC 0.97) with good feature utilization
3. **LightGBM** provides competitive results (AUC 0.96) with faster training times
4. **Decision Tree** has the lowest performance (AUC 0.89) but offers interpretability

**Feature Importance Patterns**:
- V14, V4, V10, V12, and V17 consistently rank as top predictive features across models
- `Time` and `Amount` show varying importance across different algorithms
- Ensemble methods (Random Forest, XGBoost, LightGBM) distribute importance more evenly

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation
  - `scikit-learn`: Model training, preprocessing, metrics
  - `imbalanced-learn`: SMOTE implementation
  - `xgboost`: Extreme Gradient Boosting
  - `lightgbm`: Light Gradient Boosting Machine
  - `matplotlib`, `seaborn`: Visualization
  - `joblib`: Model serialization
  - `kagglehub`: Dataset download

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── models/                          # Saved model files
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── lightgbm_model.pkl
│
├── best_fraud_detection_model.pkl   # Best performing model
├── fraud_detection.ipynb            # Main notebook
├── model_comparison.png             # Visualization plots
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm matplotlib seaborn joblib kagglehub
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Run the Jupyter notebook:
```bash
jupyter notebook fraud_detection.ipynb
```

3. The notebook will:
   - Download the dataset from Kaggle
   - Preprocess and balance the data
   - Train all four models
   - Generate comparison visualizations
   - Save the best model

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load the best model
model = joblib.load('best_fraud_detection_model.pkl')

# Make predictions
predictions = model.predict(X_new)
fraud_probability = model.predict_proba(X_new)[:, 1]
```

## 📊 Dataset Information

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions
- **Features**: 30 (28 PCA-transformed features V1-V28, Time, Amount)
- **Target**: Class (0 = legitimate, 1 = fraudulent)
- **Class Distribution**: 
  - Legitimate: 99.83%
  - Fraudulent: 0.17%

## 🔍 Methodology

1. **Data Preprocessing**:
   - Standardize `Amount` feature using StandardScaler
   - Split data (80% train, 20% test) with stratification to maintain class distribution

2. **Handling Imbalance**:
   - Apply SMOTE to training data only
   - Maintains realistic test set distribution for evaluation

3. **Model Training**:
   - Train four different algorithms with default parameters
   - Use consistent random_state=42 for reproducibility

4. **Evaluation**:
   - Generate comprehensive metrics (confusion matrix, ROC-AUC, precision-recall)
   - Compare models across multiple dimensions
   - Select and save best model based on accuracy

## 🎓 Learning Outcomes

This project demonstrates:
- Handling severely imbalanced datasets
- Comparing multiple machine learning algorithms
- Understanding ensemble methods (Random Forest, XGBoost, LightGBM)
- Evaluating models with appropriate metrics for imbalanced data
- Model persistence and deployment preparation

## 📝 Future Improvements

- [ ] Implement cross-validation for more robust evaluation
- [ ] Hyperparameter tuning using GridSearchCV or Optuna
- [ ] Try other resampling techniques (ADASYN, BorderlineSMOTE)
- [ ] Deploy best model as API using Flask/FastAPI
- [ ] Add real-time prediction capabilities
- [ ] Implement model monitoring and drift detection
- [ ] Add explainability features (SHAP values, LIME)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset provided by Kaggle and the Machine Learning Group - ULB
- Inspired by the need for robust fraud detection in financial systems
- Built as part of learning Decision Trees and ensemble methods

---

**Note**: This project is for educational purposes. In production systems, additional security measures, monitoring, and compliance considerations would be necessary.