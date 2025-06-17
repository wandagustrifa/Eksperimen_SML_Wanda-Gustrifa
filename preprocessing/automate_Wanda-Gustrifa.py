import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def preprocess_diabetes_data(df_raw, is_training=True, preprocessor_path=None):
    """
    Melakukan preprocessing pada dataset diabetes.

    Args:
        input_filepath (str): Path ke file dataset mentah.
        output_filepath (str): Path untuk menyimpan dataset yang sudah diproses.
    Returns:
        pd.DataFrame: DataFrame yang sudah diproses.
    """
    df_processed = df_raw.copy()

    # Menghapus kolom 'PatientID' dan menangani duplikat
    if 'PatientID' in df_processed.columns:
        df_processed.drop('PatientID', axis=1, inplace=True)
        print("Kolom 'PatientID' dihapus.")
    
    if df_processed.duplicated().sum() > 0:
        print(f"Ditemukan {df_processed.duplicated().sum()} duplikat. Menghapus duplikat.")
        df_processed.drop_duplicates(inplace=True)
    else:
        print("Tidak ada duplikat ditemukan.")

    # Pisahkan fitur (X) dan target (y)
    X = df_processed.drop('Diagnosis', axis=1)
    y = df_processed['Diagnosis']

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

    # Membuat pipeline preprocessing
    preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    if is_training:
        X_processed_array = preprocessing_pipeline.fit_transform(X)
        if preprocessor_path:
            joblib.dump(preprocessing_pipeline, preprocessor_path) # Simpan preprocessor
    else:
        if not preprocessor_path:
            raise ValueError("preprocessor_path harus disediakan untuk mode inferensi (is_training=False).")
        preprocessing_pipeline = joblib.load(preprocessor_path) # Muat preprocessor
        X_processed_array = preprocessing_pipeline.transform(X)

    # Mendapatkan nama kolom setelah preprocessing
    numeric_cols_transformed = numeric_features
    ohe_feature_names = preprocessing_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    categorical_cols_transformed = ohe_feature_names.get_feature_names_out(categorical_features_to_encode)
    transformed_column_names = numeric_cols_transformed + list(categorical_cols_transformed) + binary_features

    # Buat DataFrame dari data yang sudah diproses
    X_preprocessed_df = pd.DataFrame(X_processed_array, columns=transformed_column_names, index=X.index)
    
    if is_training:
        return X_preprocessed_df, y, preprocessing_pipeline
    else:
        return X_preprocessed_df, y

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        df_raw = pd.read_csv("../namadataset_raw/diabetes_data.csv") 
        print("Raw data loaded for automation test.")

        # Preprocessing untuk pelatihan dan simpan preprocessor
        X_preprocessed, y_target, preprocessor_obj = preprocess_diabetes_data(
            df_raw,
            is_training=True,
            preprocessor_path="namadataset_preprocessing/preprocessor_dataset.joblib" 
        )
        print("\nData preprocessed for training (automated).")
        print("Shape X_preprocessed:", X_preprocessed.shape)
        print("Shape y_target:", y_target.shape)
        print("Preprocessor saved.")

        # Save data
        preprocessed_data = X_preprocessed.copy()
        preprocessed_data['Diagnosis'] = y_target 
        preprocessed_data.to_csv("namadataset_preprocessing/preprocessed_diabetes_data.csv", index=False)
        print("Preprocessed data saved to namadataset_preprocessing/preprocessed_diabetes_data.csv")

        # Misalkan ada data baru yang masuk
        # df_new_raw = pd.read_csv("new_diabetes_data.csv") # Contoh data baru
        # X_new_preprocessed, y_new_target = preprocess_diabetes_data(
        #     df_new_raw,
        #     is_training=False,
        #     preprocessor_path="namadataset_preprocessing/preprocessor_dataset.joblib"
        # )
        # print("\nNew data preprocessed for inference (automated).")
        # print("Shape X_new_preprocessed:", X_new_preprocessed.shape)

    except FileNotFoundError:
        print("Error: Pastikan file diabetes_health_dataset.csv ada di folder ../namadataset_raw/.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
