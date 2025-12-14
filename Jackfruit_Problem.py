# =====================================================
# Interactive Linear / Multiple Regression (CLI)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -----------------------------
# STEP 1: Load CSV
# -----------------------------
csv_path = input("Enter path to CSV file: ")
df = pd.read_csv(csv_path)

print("\nDataset Loaded Successfully!")
print(df.head())

# -----------------------------
# STEP 2: Choose Regression Type
# -----------------------------
print("\nChoose Regression Type:")
print("1. Linear Regression")
print("2. Multiple Regression")
choice = int(input("Enter choice (1 or 2): "))

# -----------------------------
# STEP 3: Choose Target Column
# -----------------------------
print("\nAvailable columns:", list(df.columns))
target = input("Enter target column name: ")

if target not in df.columns:
    raise ValueError("Invalid target column")

# -----------------------------
# STEP 4: Feature Selection
# -----------------------------
if choice == 1:
    print("\nChoose ONE feature for Linear Regression:")
    features = [input("Enter feature column name: ")]
else:
    print("\nUsing ALL remaining columns for Multiple Regression")
    features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# -----------------------------
# STEP 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 6: Model Training
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# STEP 7: Metrics
# -----------------------------
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))

print("\n--- MODEL METRICS ---")
print(f"R-squared: {r2:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# Adjusted R² (Multiple Regression only)
if choice == 2:
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)
    print(f"Adjusted R-squared: {adj_r2:.4f}")

# -----------------------------
# STEP 8: Coefficients
# -----------------------------
print("\n--- MODEL COEFFICIENTS ---")
for col, coef in zip(features, model.coef_):
    print(f"{col}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# -----------------------------
# STEP 9: VIF (Multiple Regression)
# -----------------------------
if choice == 2:
    print("\n--- VIF (Multicollinearity Check) ---")
    X_vif = X_train.values
    for i, col in enumerate(features):
        vif = variance_inflation_factor(X_vif, i)
        print(f"{col}: {vif:.2f}")

# -----------------------------
# STEP 10: Visualizations
# -----------------------------
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# -----------------------------
# Regression Line (Linear only)
# -----------------------------
if choice == 1:
    plt.figure()
    plt.scatter(X_test[features[0]], y_test, label="Actual")
    plt.plot(X_test[features[0]], y_pred, color="red", label="Regression Line")
    plt.xlabel(features[0])
    plt.ylabel(target)
    plt.title("Linear Regression Line")
    plt.legend()
    plt.show()

# -----------------------------
# STEP 11: Residual Diagnostics
# -----------------------------
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

plt.figure()
plt.hist(residuals, bins=20)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()

# -----------------------------
# STEP 12: Conclusion
# -----------------------------
print("\n--- CONCLUSION ---")
if r2 > 0.7:
    print("Strong model performance.")
elif r2 > 0.4:
    print("Moderate model performance.")
else:
    print("Weak model performance.")
