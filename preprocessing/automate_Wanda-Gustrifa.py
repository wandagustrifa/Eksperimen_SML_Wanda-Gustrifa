import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def preprocess_diabetes_data(input_filepath, output_filepath):
    """
    Melakukan preprocessing pada dataset diabetes.

    Args:
        input_filepath (str): Path ke file dataset mentah.
        output_filepath (str): Path untuk menyimpan dataset yang sudah diproses.
    Returns:
        pd.DataFrame: DataFrame yang sudah diproses.
    """
    print(f"Memuat data dari: {input_filepath}")
    df = pd.read_csv(input_filepath)

    # Menghapus kolom 'PatientID' dan menangani duplikat
    if 'PatientID' in df.columns:
        df.drop('PatientID', axis=1, inplace=True)
        print("Kolom 'PatientID' dihapus.")
    
    if df.duplicated().sum() > 0:
        print(f"Ditemukan {df.duplicated().sum()} duplikat. Menghapus duplikat.")
        df.drop_duplicates(inplace=True)
    else:
        print("Tidak ada duplikat ditemukan.")

    # Pisahkan fitur (X) dan target (y)
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Identifikasi kolom berdasarkan tipe data dan kebutuhan preprocessing
    numeric_features = [
        'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
        'SystolicBP', 'DiastolicBP', 'FastingBloodSugar', 'HbA1c', 'SerumCreatinine',
        'BUNLevels', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
        'CholesterolTriglycerides', 'FatigueLevels', 'QualityOfLifeScore',
        'MedicalCheckupsFrequency', 'MedicationAdherence', 'HealthLiteracy'
    ]

    categorical_features_to_encode = [
        'Ethnicity', 'SocioeconomicStatus', 'EducationLevel'
    ]

    binary_features = [
        'Gender', 'Smoking', 'FamilyHistoryDiabetes', 'GestationalDiabetes',
        'PolycysticOvarySyndrome', 'PreviousPreDiabetes', 'Hypertension',
        'AntihypertensiveMedications', 'Statins', 'AntidiabeticMedications',
        'FrequentUrination', 'ExcessiveThirst', 'UnexplainedWeightLoss',
        'BlurredVision', 'SlowHealingSores', 'TinglingHandsFeet',
        'HeavyMetalsExposure', 'OccupationalExposureChemicals', 'WaterQuality'
    ]
    
    # Preprocessor menggunakan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_to_encode),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'
    )

    # Terapkan preprocessing
    X_preprocessed_array = preprocessor.fit_transform(X)

    # Mendapatkan nama kolom setelah preprocessing
    numeric_cols_transformed = numeric_features
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_to_encode)
    categorical_cols_transformed = list(ohe_feature_names)
    transformed_column_names = numeric_cols_transformed + categorical_cols_transformed + binary_features

    X_preprocessed_df = pd.DataFrame(X_preprocessed_array, columns=transformed_column_names)
    
    y_reset_index = y.reset_index(drop=True)
    df_preprocessed = pd.concat([X_preprocessed_df, y_reset_index], axis=1)

    # Pastikan direktori output ada
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Direktori '{output_dir}' dibuat.")

    # Simpan data yang sudah diproses
    df_preprocessed.to_csv(output_filepath, index=False)
    print(f"Data yang sudah diproses berhasil disimpan ke: {output_filepath}")
    print(f"Bentuk data yang sudah diproses: {df_preprocessed.shape}")
    
    return df_preprocessed

if __name__ == "__main__":
    raw_data_path = os.path.join(current_script_dir, '..', 'namadataset_raw', 'diabetes_data.csv')
    preprocessed_data_path = os.path.join(current_script_dir, 'namadataset_preprocessing', 'preprocessed_diabetes_data.csv')

    _ = preprocess_diabetes_data(raw_data_path, preprocessed_data_path)
