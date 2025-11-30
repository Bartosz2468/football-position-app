import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- ŁADOWANIE MODELU ---
model = joblib.load("rf_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("⚽ Predykcja optymalnej pozycji piłkarskiej")
st.write("Wprowadź swoje dane, aby otrzymać dopasowanie pozycji z TOP 3 rankingiem.")

# --- FORMULARZ ---
acc = st.slider("Acceleration", 1, 99, 50)
spr = st.slider("Sprint Speed", 1, 99, 50)
agi = st.slider("Agility", 1, 99, 50)
bal = st.slider("Balance", 1, 99, 50)
jmp = st.slider("Jumping", 1, 99, 50)
sta = st.slider("Stamina", 1, 99, 50)
strg = st.slider("Strength", 1, 99, 50)

foot = st.radio("Preferred foot", ["Right", "Left"])
foot_val = 1 if foot == "Right" else 0

height = st.number_input("Height (cm)", 140, 220, 180)
weight = st.number_input("Weight (kg)", 40, 120, 75)
age = st.number_input("Age", 10, 50, 20)

# --- PRZYCISK ---
if st.button("Oblicz pozycję"):

    # Przygotowanie danych
    new_player = pd.DataFrame([{
        'Acceleration': acc,
        'Sprint Speed': spr,
        'Agility': agi,
        'Balance': bal,
        'Jumping': jmp,
        'Stamina': sta,
        'Strength': strg,
        'Preferred foot': foot_val,
        'Height': height,
        'Weight': weight,
        'Age': age
    }])

    # RF NIE wymaga skalowania — używamy danych bezpośrednio
    probs = model.predict_proba(new_player)[0]

    # TOP3 indeksy
    top3_idx = probs.argsort()[-3:][::-1]

    st.subheader("🏆 TOP 3 dopasowania:")

    labels = []
    values = []

    for idx in top3_idx:
        pos = encoder.inverse_transform([idx])[0]
        val = probs[idx] * 100
        labels.append(pos)
        values.append(probs[idx])
        st.write(f"**{pos}** — {val:.2f}%")

    # --- WYKRES SŁUPKOWY ---
    chart_data = pd.DataFrame({
        "Pozycja": labels,
        "Prawdopodobieństwo": values
    }).set_index("Pozycja")

    st.subheader("📊 Wykres dopasowania:")
    st.bar_chart(chart_data)
