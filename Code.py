# MACHINE LEARNING MODEL 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. LOAD DATA
df = pd.read_csv("student_perform.csv")   # change if filename differs

# Clean column names
df.columns = df.columns.str.strip()

print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# 2. DATA PREPROCESSING

# Remove missing values
df = df.dropna()

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print("\nAfter Encoding:")
print(df.head())

# 3. FEATURE SELECTION
# Target variable (Final Grade)
target = 'G3'

X = df.drop(columns=[target])
y = df[target]

print("\nSelected Features:")
print(X.columns)

# 4. FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. MODEL TRAINING

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 7. MODEL EVALUATION

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n===== MODEL PERFORMANCE =====")
print(f"Linear Regression RMSE: {rmse_lr:.2f}")
print(f"Random Forest RMSE: {rmse_rf:.2f}")

# 8. VISUALIZATION

plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted (Random Forest)")
plt.savefig("prediction_plot.png")
plt.close()

# 9. SAVE OUTPUT

output = pd.DataFrame({
    "Actual_G3": y_test,
    "LR_Predicted": y_pred_lr,
    "RF_Predicted": y_pred_rf
})

output.to_csv("ml_output.csv", index=False)

print("\nFiles Created:")
print("1. ml_output.csv (predictions)")
print("2. prediction_plot.png (graph)")



if rmse_rf < rmse_lr:
    print("\nRandom Forest performed better.")
else:
    print("\nLinear Regression performed better.")