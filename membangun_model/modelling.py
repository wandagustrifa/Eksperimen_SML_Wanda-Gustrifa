import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil 
import json
import dagshub

dagshub.init(repo_owner='wandagustrifa', repo_name='diabetes-mlops-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/wandagustrifa/diabetes-mlops-project.mlflow") 

def plot_confusion_matrix_and_save(y_true, y_pred, labels, output_folder="temp_plots", filename="training_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    os.makedirs(output_folder, exist_ok=True)
    full_filepath = os.path.join(output_folder, filename)
    plt.savefig(full_filepath)
    plt.close()
    print(f"Confusion Matrix disimpan sementara ke: {full_filepath}")
    return full_filepath

# --- FUNGSI BANTUAN: plot_roc_curve_and_save ---
def plot_roc_curve_and_save(y_true, y_pred_proba, filename="roc_curve.png", output_folder="temp_plots"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    os.makedirs(output_folder, exist_ok=True)
    full_filepath = os.path.join(output_folder, filename)
    plt.savefig(full_filepath)
    plt.close()
    print(f"ROC Curve disimpan sementara ke: {full_filepath}")
    return full_filepath

# --- FUNGSI BANTUAN: create_dummy_html_json_files ---
def create_dummy_html_json_files(output_folder, estimator_html_filename="estimator.html", metric_info_json_filename="metric_info.json"):
    os.makedirs(output_folder, exist_ok=True)

    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Model Estimator Info</title></head>
    <body>
        <h1>Logistic Regression Model Overview</h1>
        <p>This is a placeholder for detailed estimator visualization (e.g., from skompiler or custom diagram).</p>
        <pre>
        LogisticRegression()
        </pre>
    </body>
    </html>
    """
    html_filepath = os.path.join(output_folder, estimator_html_filename)
    with open(html_filepath, "w") as f:
        f.write(html_content)
    print(f"Dummy estimator.html disimpan sementara ke: {html_filepath}")

    json_content = {
        "metrics": {
            "accuracy": "See logged metrics",
            "f1_score": "See logged metrics",
            "additional_info": "This JSON can contain more raw metric data or run details."
        },
        "model_type": "Logistic Regression",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    json_filepath = os.path.join(output_folder, metric_info_json_filename)
    with open(json_filepath, "w") as f:
        json.dump(json_content, f, indent=4)
    print(f"Dummy metric_info.json disimpan sementara ke: {json_filepath}")
    
    return html_filepath, json_filepath

def train_model(input_filepath):
    """
    Memuat data yang sudah diproses, melatih model Logistic Regression,
    dan melacak metrik serta artefak menggunakan MLflow.

    Args:
        input_filepath (str): Path ke file dataset yang sudah diproses.
    """
    print(f"Memuat data yang sudah diproses dari: {input_filepath}")
    df_preprocessed = pd.read_csv(input_filepath)

    # Pisahkan fitur (X) dan target (y)
    X = df_preprocessed.drop('Diagnosis', axis=1)
    y = df_preprocessed['Diagnosis']

    # Pembagian Data Pelatihan dan Pengujian
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        # Definisikan model
        model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)

        # Latih model
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Hitung metrik evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print("\nMetrik Model:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")

        # Log parameter
        mlflow.log_param("solver", 'liblinear')
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)

        # Log metrik secara manual (sesuai kriteria Skilled)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)
        
        # Metrik tambahan seperti Specificity dan False Positive Rate
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) != 0 else 0
        mlflow.log_metric("specificity", specificity)
        mlflow.log_metric("false_positive_rate", false_positive_rate)
        temp_artifacts_dir = "temp_mlflow_artifacts" 
        os.makedirs(temp_artifacts_dir, exist_ok=True)

        # Artefak 1: Confusion Matrix Plot
        print("\nMelog Confusion Matrix sebagai Artefak...")
        class_labels = ['No Diabetes', 'Diabetes']
        cm_path = plot_confusion_matrix_and_save(
            y_test, y_pred, class_labels, output_folder=temp_artifacts_dir, filename="training_confusion_matrix.png"
        )
        # Log ke root artefak run (artifact_path="")
        mlflow.log_artifact(cm_path, artifact_path="")

        # Artefak 2: ROC Curve Plot
        print("Melog ROC Curve sebagai Artefak...")
        roc_path = plot_roc_curve_and_save(
            y_test, y_pred_proba, output_folder=temp_artifacts_dir, filename="roc_curve.png"
        )
        # Log ke root artefak run (artifact_path="")
        mlflow.log_artifact(roc_path, artifact_path="")

        # Artefak 3: estimator.html
        print("Melog estimator.html dummy sebagai Artefak...")
        html_path, _ = create_dummy_html_json_files( # _ untuk mengabaikan return kedua
            output_folder=temp_artifacts_dir,
            estimator_html_filename="estimator.html",
            metric_info_json_filename="metric_info_temp.json" # Nama file JSON sementara untuk menghindari konflik
        )
        # Log ke root artefak run (artifact_path="")
        mlflow.log_artifact(html_path, artifact_path="")

        # Artefak 4: metric_info.json
        print("Melog metric_info.json dummy sebagai Artefak...")
        _, json_path = create_dummy_html_json_files( # _ untuk mengabaikan return pertama
            output_folder=temp_artifacts_dir,
            estimator_html_filename="estimator_temp.html", # Nama file HTML sementara untuk menghindari konflik
            metric_info_json_filename="metric_info.json"
        )
        # Log ke root artefak run (artifact_path="")
        mlflow.log_artifact(json_path, artifact_path="")


        # Log Model sebagai Artefak MLflow
        # Ini akan membuat folder "model" di root direktori artefak run.
        mlflow.sklearn.log_model(model, "logistic_regression_model") # Nama folder artefak model

        # Bersihkan folder sementara setelah dilog
        shutil.rmtree(temp_artifacts_dir)
        print("File artefak sementara telah dihapus.")

        print("\nMLflow run for tuned model completed.")

if __name__ == "__main__":
    preprocessed_data_path = 'namadataset_preprocessing/preprocessed_diabetes_data.csv'
    train_model(preprocessed_data_path)