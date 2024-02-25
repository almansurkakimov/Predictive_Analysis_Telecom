from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

columns = [f"feature_{i+1}" for i in range(X.shape[1])]
data = pd.DataFrame(X, columns=columns)
data['churn'] = y

data.to_csv('telecom_churn_dataset.csv', index=False)