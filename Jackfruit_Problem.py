# ============================================================
# Advanced Linear / Multiple Regression Tool (UI + Plots)
# ============================================================

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -----------------------------
# Global dataset
# -----------------------------
df = None

# -----------------------------
# Load CSV
# -----------------------------
def load_csv():
    global df
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not path:
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    cols_list.delete(0, tk.END)
    for col in df.columns:
        cols_list.insert(tk.END, col)

    messagebox.showinfo("Success", "CSV file loaded successfully")

# -----------------------------
# Run Regression
# -----------------------------
def run_regression():
    global df
    output.delete("1.0", tk.END)

    if df is None:
        messagebox.showerror("Error", "Load a CSV file first")
        return

    target = target_var.get().strip()
    if target not in df.columns:
        messagebox.showerror("Error", "Invalid target column")
        return

    regression_type = reg_type.get()

    # -----------------------------
    # Feature selection
    # -----------------------------
    if regression_type == 1:
        feature = feature_var.get().strip()
        if feature not in df.columns:
            messagebox.showerror("Error", "Invalid feature column")
            return
        X = df[[feature]]
    else:
        X = df.drop(columns=[target])
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    y = df[target]

    # Clean data
    data = pd.concat([X, y], axis=1).dropna()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if X.shape[0] < 5:
        messagebox.showerror("Error", "Insufficient data after cleaning")
        return

    # -----------------------------
    # Train/Test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Train model
    # -----------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))

    output.insert(tk.END, "📌 MODEL PERFORMANCE\n")
    output.insert(tk.END, "-" * 40 + "\n")
    output.insert(tk.END, f"R² Score : {r2:.4f}\n")
    output.insert(tk.END, f"MSE      : {mse:.4f}\n")
    output.insert(tk.END, f"RMSE     : {rmse:.4f}\n")
    output.insert(tk.END, f"MAE      : {mae:.4f}\n\n")

    # -----------------------------
    # Coefficients
    # -----------------------------
    output.insert(tk.END, "📌 MODEL COEFFICIENTS\n")
    output.insert(tk.END, "-" * 40 + "\n")

    for col, coef in zip(X.columns, model.coef_):
        output.insert(tk.END, f"{col:<25} {coef:.4f}\n")

    output.insert(tk.END, f"\nIntercept: {model.intercept_:.4f}\n\n")

    # -----------------------------
    # VIF
    # -----------------------------
    if regression_type == 2:
        output.insert(tk.END, "📌 VARIANCE INFLATION FACTOR (VIF)\n")
        output.insert(tk.END, "-" * 40 + "\n")

        if X_train.shape[1] < 2:
            output.insert(tk.END, "VIF not applicable (only one predictor)\n\n")
        else:
            X_vif = X_train.loc[:, X_train.var() != 0]
            if X_vif.shape[1] < 2:
                output.insert(tk.END, "VIF not applicable (constant predictors)\n\n")
            else:
                X_vif = sm.add_constant(X_vif)
                for i, col in enumerate(X_vif.columns):
                    vif = variance_inflation_factor(X_vif.values, i)
                    output.insert(tk.END, f"{col:<25} {vif:.2f}\n")
                output.insert(tk.END, "\n")

    # -----------------------------
    # PLOTS
    # -----------------------------

    # 1️⃣ Correlation Heatmap
    plt.figure()
    corr = df.select_dtypes(include=np.number).corr()
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.show()

    # 2️⃣ Actual vs Predicted
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

    # 3️⃣ Residuals vs Predicted
    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.show()

    # 4️⃣ Residual Distribution
    plt.figure()
    plt.hist(residuals, bins=20)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.show()

    # 5️⃣ Q-Q Plot
    sm.qqplot(residuals, line="45")
    plt.title("Q-Q Plot of Residuals")
    plt.show()

    # 6️⃣ Feature Importance
    plt.figure()
    coef_abs = np.abs(model.coef_)
    plt.barh(X.columns, coef_abs)
    plt.xlabel("Absolute Coefficient Value")
    plt.title("Feature Importance")
    plt.show()

    # 7️⃣ Linear Regression Line
    if regression_type == 1:
        plt.figure()
        plt.scatter(X_test[feature], y_test)
        plt.plot(X_test[feature], y_pred)
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title("Linear Regression Fit")
        plt.show()

# -----------------------------
# UI DESIGN (MODERN DASHBOARD)
# -----------------------------

root = tk.Tk()
root.title("📊 Regression Analysis Studio")
root.geometry("1100x700")
root.configure(bg="#eef1f5")

# -----------------------------
# Layout Frames
# -----------------------------
sidebar = tk.Frame(root, bg="#1f2933", width=260)
sidebar.pack(side=tk.LEFT, fill=tk.Y)

main = tk.Frame(root, bg="#eef1f5")
main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# -----------------------------
# Sidebar Content
# -----------------------------
tk.Label(
    sidebar,
    text="Regression\nStudio",
    font=("Segoe UI", 20, "bold"),
    fg="white",
    bg="#1f2933",
    justify="left"
).pack(pady=30, padx=20, anchor="w")

tk.Button(
    sidebar,
    text="📂  Load CSV",
    command=load_csv,
    font=("Segoe UI", 11),
    bg="#2563eb",
    fg="white",
    relief="flat",
    padx=10,
    pady=8
).pack(padx=20, fill=tk.X)

tk.Label(
    sidebar,
    text="Dataset Columns",
    fg="#9ca3af",
    bg="#1f2933",
    font=("Segoe UI", 10)
).pack(pady=(30, 5), padx=20, anchor="w")

cols_list = tk.Listbox(
    sidebar,
    height=15,
    bg="#111827",
    fg="white",
    relief="flat",
    highlightthickness=0
)
cols_list.pack(padx=20, fill=tk.BOTH, expand=True)

# -----------------------------
# Header
# -----------------------------
header = tk.Frame(main, bg="#eef1f5")
header.pack(fill=tk.X, pady=15, padx=20)

tk.Label(
    header,
    text="Linear & Multiple Regression Analysis",
    font=("Segoe UI", 18, "bold"),
    bg="#eef1f5"
).pack(anchor="w")

tk.Label(
    header,
    text="Model diagnostics, statistical plots & performance metrics",
    font=("Segoe UI", 11),
    fg="#4b5563",
    bg="#eef1f5"
).pack(anchor="w")

# -----------------------------
# Control Card
# -----------------------------
control_card = tk.Frame(main, bg="white", padx=20, pady=20)
control_card.pack(fill=tk.X, padx=20, pady=10)

tk.Label(control_card, text="Target Column", bg="white").grid(row=0, column=0, sticky="w")
target_var = tk.StringVar()
tk.Entry(control_card, textvariable=target_var, width=25).grid(row=1, column=0, padx=5)

tk.Label(control_card, text="Feature (Linear Only)", bg="white").grid(row=0, column=1, sticky="w")
feature_var = tk.StringVar()
tk.Entry(control_card, textvariable=feature_var, width=25).grid(row=1, column=1, padx=5)

reg_type = tk.IntVar(value=1)
tk.Radiobutton(
    control_card, text="Linear Regression",
    variable=reg_type, value=1, bg="white"
).grid(row=1, column=2, padx=10)

tk.Radiobutton(
    control_card, text="Multiple Regression",
    variable=reg_type, value=2, bg="white"
).grid(row=1, column=3, padx=10)

tk.Button(
    control_card,
    text="▶ Run Analysis",
    command=run_regression,
    bg="#10b981",
    fg="white",
    font=("Segoe UI", 11, "bold"),
    relief="flat",
    padx=15,
    pady=8
).grid(row=1, column=4, padx=15)

# -----------------------------
# Output Card
# -----------------------------
output_card = tk.Frame(main, bg="white", padx=20, pady=20)
output_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 20))

tk.Label(
    output_card,
    text="Analysis Output",
    font=("Segoe UI", 14, "bold"),
    bg="white"
).pack(anchor="w", pady=(0, 10))

output = ScrolledText(
    output_card,
    font=("Consolas", 10),
    bg="#0f172a",
    fg="#e5e7eb",
    insertbackground="white",
    relief="flat"
)
output.pack(fill=tk.BOTH, expand=True)

root.mainloop()
