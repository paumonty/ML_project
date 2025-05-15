import streamlit as st
import pandas as pd
import joblib
import os

modelo = joblib.load("../4. Models/modelo_entrenado.pkl")


st.set_page_config(page_title="Predicción de Préstamos", layout="centered")
st.title("📊 Predicción de Riesgo de Impago")
st.write("Introduce los datos del cliente:")


# === Inputs distribuidos en columnas ===
st.markdown("### 📋 Introduce los datos del cliente")

col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Edad del cliente", 18, 75, 30)
    genero = st.selectbox("Género", ['male', 'female'])
    educacion = st.selectbox("Nivel educativo", ['High School', 'Bachelor', 'Master'])
    experiencia = st.slider("Experiencia laboral (años)", 0, 40, 5)
    vivienda = st.selectbox("Tipo de vivienda", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])

with col2:
    ingreso = st.number_input("Ingresos mensuales (€)", min_value=500, value=2000)
    prestamo = st.number_input("Importe del préstamo (€)", min_value=500, value=5000)
    intencion = st.selectbox("Finalidad del préstamo", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    score = st.slider("Puntaje de crédito", 300, 850, 600)
    impagos = st.radio("¿Impagos anteriores?", ['No', 'Yes'])

# === Evaluación del riesgo ===
if st.button("📊 Evaluar riesgo"):
    datos_cliente = pd.DataFrame([{
        'person_age': int(edad),
        'person_gender': str(genero),
        'person_education': str(educacion),
        'person_income': float(ingreso),
        'person_emp_exp': int(experiencia),
        'person_home_ownership': str(vivienda),
        'loan_amnt': float(prestamo),
        'loan_intent': str(intencion),
        'credit_score': int(score),
        'previous_loan_defaults_on_file': str(impagos)
    }])

    # Mostrar los datos ingresados
    st.markdown("### 🧾 Datos ingresados")
    st.dataframe(datos_cliente)

    # Evaluación del modelo con spinner de carga
    with st.spinner("Evaluando el riesgo..."):
        pred = modelo.predict(datos_cliente)[0]
        prob = modelo.predict_proba(datos_cliente)[0][1]

    # Mostrar resultado
    st.markdown("### 📈 Resultado de la evaluación")
    if pred == 1:
        st.success(f"✅ **Riesgo Bajo** — Probabilidad de pago: {prob:.2%}")
    else:
        st.error(f"⚠️ **Riesgo Alto** — Probabilidad de impago: {(1 - prob):.2%}")

# === Footer ===
st.markdown("""
---
Desarrollado con ❤️ por **Pau Montañés Pla**  
📧 Contacto: paumontanes@ejemplo.com  
""")

