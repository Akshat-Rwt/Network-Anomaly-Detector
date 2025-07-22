import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_mse_distribution(mse, threshold):
    fig, ax = plt.subplots()
    ax.hist(mse, bins=50, alpha=0.7)
    ax.axvline(threshold, color='red', linestyle='--', label=f"Threshold: {threshold:.5f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Frequency")
    ax.set_title("Autoencoder Reconstruction Error")
    ax.legend()
    return fig

def plot_mse_distribution(mse, threshold):
    fig, ax = plt.subplots()
    ax.hist(mse, bins=50, alpha=0.6, label='MSE')
    ax.axvline(threshold, color='red', linestyle='--', label='Threshold')
    ax.set_title('MSE Distribution')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Frequency')
    ax.legend()
    return fig

def generate_result_df(preds_if, preds_ae, y_true):
    return pd.DataFrame({
        'IsolationForest': preds_if,
        'Autoencoder': preds_ae,
        'GroundTruth': y_true
    })

def plot_anomaly_counts(count_if, count_ae):
    fig, ax = plt.subplots()
    ax.bar(['Isolation Forest', 'Autoencoder'], [count_if, count_ae], color=['blue', 'green'])
    ax.set_title('Anomaly Counts by Model')
    ax.set_ylabel('Count')
    return fig

def plot_pie_chart(preds, title):
    normal = preds.count(0)
    anomaly = preds.count(1)
    fig, ax = plt.subplots()
    ax.pie([normal, anomaly], labels=['Normal', 'Anomaly'], autopct='%1.1f%%', colors=['lightblue', 'red'])
    ax.set_title(title)
    return fig

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def boxplot_feature_by_anomaly(df, preds, feature_name):
    df_plot = df.copy()
    df_plot['Anomaly'] = preds
    fig, ax = plt.subplots()
    sns.boxplot(data=df_plot, x='Anomaly', y=feature_name, ax=ax)
    ax.set_title(f'{feature_name} Distribution by Anomaly')
    return fig

def generate_result_df(preds_if, preds_ae, y):
    return pd.DataFrame({
        "IsolationForest": preds_if,
        "Autoencoder": preds_ae,
        "ActualLabel": y
    })
