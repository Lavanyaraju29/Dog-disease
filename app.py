import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
disease_data = pd.read_csv('projectdataset.csv', encoding='ISO-8859-1')
disease_data = disease_data.dropna(subset=['disease', 'part', 'severity'])

# Prepare data
symptom_columns = [col for col in disease_data.columns if 'Symptoms' in col]
disease_data[symptom_columns] = disease_data[symptom_columns].fillna('')
disease_data['Combined_Symptoms'] = disease_data[symptom_columns].agg(' '.join, axis=1)
disease_data['part'] = disease_data['part'].str.lower().apply(lambda x: x.rstrip('s'))
disease_data['Combined_Symptoms'] = disease_data['Combined_Symptoms'].str.lower()
disease_data['severity'] = disease_data['severity'].str.lower()

# Helper functions
def normalize_part(part):
    return part.lower().rstrip('s')

def map_severity(severity):
    severity = severity.lower()
    if severity in ['low', 'mild']:
        return ['low', 'mild']
    elif severity in ['medium', 'moderate']:
        return ['medium', 'moderate']
    elif severity in ['high', 'severe']:
        return ['high', 'severe']
    return [severity]

def ml_diagnose_disease(part, symptoms, severity):
    part = normalize_part(part)
    severity_options = map_severity(severity)
    part_filtered_data = disease_data[
        (disease_data['part'] == part) &
        (disease_data['severity'].isin(severity_options))
    ]
    
    if part_filtered_data.empty:
        return "No diseases found for this body part with the specified severity level.", None

    X = part_filtered_data['part'] + ' ' + part_filtered_data['Combined_Symptoms']
    y = part_filtered_data['disease']
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    input_text = part + ' ' + ' '.join(symptoms).lower()
    input_vec = vectorizer.transform([input_text])
    prediction = model.predict(input_vec)[0]

    return prediction, vectorizer

def recommend_treatment(disease):
    row = disease_data[disease_data['disease'] == disease]
    if not row.empty:
        return row['treatment'].values[0]
    return "No specific treatment found."

# --- Streamlit UI ---
st.title("🐶 Dog Disease Diagnosis Chatbot")

# Input for affected body part
body_parts = sorted(disease_data['part'].unique())
part_input = st.selectbox("Select affected body part", body_parts)

# Input for severity
severity_input = st.selectbox("Select severity level", ['low', 'medium', 'high'])

# Input for symptoms
symptoms_input = st.text_input("Enter symptoms (comma-separated)", "")

if st.button("Diagnose"):
    if symptoms_input.strip() == "":
        st.warning("Please enter at least one symptom.")
    else:
        symptoms = [s.strip().lower() for s in symptoms_input.split(',')]
        predicted_disease, _ = ml_diagnose_disease(part_input, symptoms, severity_input)

        if predicted_disease == "No diseases found for this body part with the specified severity level.":
            st.error(predicted_disease)
        else:
            treatment = recommend_treatment(predicted_disease)
            st.success(f"Predicted Disease: **{predicted_disease}**")
            st.info(f"Recommended Treatment: **{treatment}**")
