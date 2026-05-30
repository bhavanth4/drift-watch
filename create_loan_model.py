import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# 1. Define training data (Features: Age, Income | Target: Approved)
X_train = np.array([
    [25, 50000], 
    [45, 120000], 
    [30, 80000], 
    [22, 20000]
])
y_train = np.array([0, 1, 1, 0]) # 0 = Denied, 1 = Approved

# 2. Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Save it as a binary .joblib file
joblib.dump(model, "my_loan_model.joblib")
print("Model successfully saved as my_loan_model.joblib!")
