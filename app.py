# Day 81 - Boston House Price Prediction (Offline, Full Code)

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Step 2: Create Boston dataset locally (hardcoded from sklearn 1.1)
# Columns: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
from sklearn.datasets import fetch_openml

# Fetch Boston dataset via OpenML (offline-friendly)
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# Save locally
df.to_csv("boston.csv", index=False)
print("boston.csv created locally with shape:", df.shape)

# Step 3: Load the CSV locally
df = pd.read_csv("boston.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Step 4: Define features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Fit multivariable linear regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 9: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

# Step 10: Residual analysis
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Predicted vs Residuals")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.show()

# Step 11: Data transformation to improve model
pt = PowerTransformer()
X_train_pt = pt.fit_transform(X_train)
X_test_pt = pt.transform(X_test)

model_pt = LinearRegression()
model_pt.fit(X_train_pt, y_train)
y_pred_pt = model_pt.predict(X_test_pt)
print(f"R^2 after Power Transformation: {r2_score(y_test, y_pred_pt):.2f}")

# Step 12: Predict a new property price
# Example: [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
new_house = np.array([[0.1, 18, 2.3, 0, 0.5, 6.5, 35, 5, 4, 300, 15, 390, 5]])
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)
print(f"Estimated house price for new property: ${predicted_price[0]*1000:.2f}")
