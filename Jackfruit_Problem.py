# Imported modules which let us use multiple regression to analyse traffic 
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


#We used the following metrics to deternmine whether the data provided would give us a good fit
def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred, squared=True)
    except:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

    return r2, mae, mse, rmse



# This is the main code which
root = tk.Tk()
root.title("Traffic Analyser")
root.geometry("780x520")
root.config(bg="#24252a")


title = tk.Label(root,text="🚦 TRAFFIC ANALYSER 🚦", font=("Comic Sans MS",26,"bold"), fg="white", bg="#24252a")
title.pack(pady=20)



# ==========================================================
#                  RANDOM TEST MODE
# ==========================================================
def run_random():
    y_true=np.array([120,140,160,200,220,240,260])
    y_pred=np.array([118,145,158,198,225,238,255])

    r2,mae,mse,rmse = metrics(y_true,y_pred)

    result_box.delete("1.0","end")
    result_box.insert("end","📊 RANDOM MODE RESULT\n\n")
    result_box.insert("end",f"R2 Score : {r2:.4f}\n")
    result_box.insert("end",f"MAE      : {mae:.4f}\n")
    result_box.insert("end",f"MSE      : {mse:.4f}\n")
    result_box.insert("end",f"RMSE     : {rmse:.4f}\n\n")
    result_box.insert("end","Meaning:\n• R² close to 1 → Good fit\n• RMSE lower → Better\n• MAE shows average error\n")

    # PLOTS
    plt.figure(figsize=(7,4))
    plt.title("Actual vs Predicted")
    plt.plot(y_true,marker="o",label="Actual")
    plt.plot(y_pred,marker="x",linestyle="--",label="Predicted")
    plt.legend(); plt.grid(); plt.show()

    plt.figure(figsize=(7,4))
    plt.bar(["R2","MAE","MSE","RMSE"],[r2,mae,mse,rmse])
    plt.title("Model Metrics")
    plt.grid(axis="y")
    plt.show()

    residual = y_true - y_pred
    plt.figure(figsize=(7,4))
    plt.plot(residual,marker="o")
    plt.axhline(0,linestyle="--")
    plt.title("Residual Error")
    plt.grid(); plt.show()

    plt.figure(figsize=(6,5))
    plt.scatter(y_true,y_pred)
    mn,mx=min(y_true),max(y_true)
    plt.plot([mn,mx],[mn,mx],linestyle="--")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title("Correlation Plot")
    plt.grid(); plt.show()



# ==========================================================
#                  CUSTOM INPUT MODE
# ==========================================================
input_frame = tk.Frame(root,bg="#1e1f22")
input_frame.pack(pady=8,fill="x")

labels=["Cars","Bikes","Trucks","Hour","Weather","Traffic Flow"]
entries=[]

for i,t in enumerate(labels):
    lbl=tk.Label(input_frame,text=t,font=("Arial",11,"bold"),fg="white",bg="#1e1f22")
    lbl.grid(row=0,column=i,padx=4,pady=3)
    e=tk.Entry(input_frame,width=10,font=("Arial",11))
    e.grid(row=1,column=i,padx=4)
    entries.append(e)


data=[]

def add_row():
    try:
        row=[float(e.get()) for e in entries]
        data.append(row)

        table.insert("",tk.END,values=row)

        for e in entries: e.delete(0,'end')

    except:
        messagebox.showerror("Error","Enter numeric values only!")


def train_model():
    if len(data)<3:
        messagebox.showwarning("Not Enough Data","Enter at least 3 rows!")
        return

    df=pd.DataFrame(data,columns=["Cars","Bikes","Trucks","Hour","Weather","TrafficFlow"])
    X=df[["Cars","Bikes","Trucks","Hour","Weather"]]
    y=df["TrafficFlow"]

    model=LinearRegression()
    model.fit(X,y)
    pred=model.predict(X)

    r2,mae,mse,rmse = metrics(y,pred)

    result_box.delete("1.0","end")
    result_box.insert("end","🧠 CUSTOM MODEL TRAINED\n\n")
    result_box.insert("end",f"R2 Score : {r2:.4f}\nMAE      : {mae:.4f}\nMSE      : {mse:.4f}\nRMSE     : {rmse:.4f}\n\n")
    result_box.insert("end","📎 Feature Importance:\n")
    for f,c in zip(X.columns,model.coef_):
        result_box.insert("end",f"{f} → {c:.3f}\n")
    result_box.insert("end",f"\nBase Intercept: {model.intercept_:.3f}")

    # PLOTS
    plt.figure(figsize=(7,4))
    plt.plot(y,marker="o",label="Actual")
    plt.plot(pred,marker="x",linestyle="--",label="Predicted")
    plt.legend(); plt.grid(); plt.show()

    residual=y-pred
    plt.figure(figsize=(7,4))
    plt.plot(residual,marker="o")
    plt.axhline(0,linestyle="--")
    plt.title("Residual Error Plot")
    plt.grid(); plt.show()

    plt.figure(figsize=(6,5))
    plt.scatter(y,pred)
    mn,mx=min(y),max(y)
    plt.plot([mn,mx],[mn,mx],linestyle="--")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Scatter")
    plt.grid(); plt.show()



# ================= TABLE =================
table=ttk.Treeview(root,columns=labels,show="headings",height=5)
for c in labels: table.heading(c,text=c)
table.pack(pady=10)



# ================= BUTTONS =================
btn_frame=tk.Frame(root,bg="#24252a")
btn_frame.pack(pady=10)

tk.Button(btn_frame,text="Run Random Mode",bg="#ffc107",font=("Comic Sans MS",14,"bold"),
          fg="black",width=15,command=run_random).grid(row=0,column=0,padx=10)

tk.Button(btn_frame,text="Add Row",bg="#00e676",font=("Comic Sans MS",14,"bold"),
          fg="black",width=15,command=add_row).grid(row=0,column=1,padx=10)

tk.Button(btn_frame,text="Train Regression",bg="#40c4ff",font=("Comic Sans MS",14,"bold"),
          fg="black",width=15,command=train_model).grid(row=0,column=2,padx=10)



# ================= RESULT BOX =================
result_box=tk.Text(root,width=70,height=8,font=("Consolas",12),bg="#101012",fg="white")
result_box.pack(pady=15)



root.mainloop()
