import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("modelo_prestamos.pkl")

genero_map = {'male': 1, 'female': 0}
educacion_map = {'High School': 0, 'Bachelor': 1, 'Master': 2}
vivienda_map = {'RENT': 2, 'OWN': 1, 'MORTGAGE': 0}
intencion_map = {'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 2, 'PERSONAL': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
impago_map = {'Yes': 1, 'No': 0}

st.set_page_config(page_title="Predicción de Préstamos", layout="centered")
st.title("Predicción de riesgo de impago")
st.write("Introduce los datos del cliente:")

edad = st.slider("Edad del cliente", 18, 75, 35)
genero = st.selectbox("Género", list(genero_map.keys()))
educacion = st.selectbox("Nivel educativo", list(educacion_map.keys()))
ingresos = st.number_input("Ingresos mensuales (€)", min_value=0, value=2000)
experiencia = st.slider("Años de experiencia laboral", 0, 40, 5)
vivienda = st.selectbox("Tipo de vivienda", list(vivienda_map.keys()))
prestamo = st.number_input("Importe del préstamo (€)", min_value=0, value=5000)
intencion = st.selectbox("Finalidad del préstamo", list(intencion_map.keys()))
score = st.slider("Puntaje de crédito", 300, 850, 600)
impagos = st.radio("¿Tiene impagos anteriores?", list(impago_map.keys()))

if st.button("Evaluar riesgo"):
    datos_cliente = pd.DataFrame([{
        'person_age': edad,
        'person_gender': genero_map[genero],
        'person_education': educacion_map[educacion],
        'person_income': ingresos,
        'person_emp_exp': experiencia,
        'person_home_ownership': vivienda_map[vivienda],
        'loan_amnt': prestamo,
        'loan_intent': intencion_map[intencion],
        'credit_score': score,
        'previous_loan_defaults_on_file': impago_map[impagos]
    }])

    prediccion = modelo.predict(datos_cliente)

    if prediccion[0] == 1:
        st.success("✅ El cliente probablemente devolverá el préstamo.")
    else:
        st.error("❌ El cliente tiene un alto riesgo de impago.")

