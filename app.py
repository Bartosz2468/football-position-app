import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- ŁADOWANIE MODELU ---
model = joblib.load("rf_model.pkl")
encoder = joblib.load("encoder.pkl")

st.title("⚽ Predykcja optymalnej pozycji piłkarskiej")

st.write("Wprowadź wyniki swoich testów")

# --- FUNKCJE PRZELICZAJĄCE ---
def scale(x, min_val, max_val):
    return int(1 + 98 * (x - min_val) / (max_val - min_val))

def scale_inverse(x, min_val, max_val):
    return int(1 + 98 * (max_val - x) / (max_val - min_val))


# --- FORMULARZ REALNYCH TESTÓW ---
st.header("Testy sprawnościowe")

t10 = st.number_input("Czas biegu na 10 m (sekundy)", 1.5, 4.0, 2.0)
t30 = st.number_input("Czas biegu na 30 m (sekundy)", 3.3, 7.0, 4.2)
t_test = st.number_input("T-test agility (sekundy)", 8.0, 20.0, 11.0)
balance = st.number_input("Stanie na 1 nodze (sekundy)", 1, 60, 20)
cmj = st.number_input("Wyskok pionowy CMJ (cm)", 10, 100, 40)
coop = st.number_input("Test Coopera – dystans (m)", 1000, 4000, 2500)
grip = st.number_input("Siła ścisku dłoni (kg)", 10, 80, 40)

st.header("Dane antropometryczne")

foot = st.radio("Preferred foot", ["Right", "Left"])
foot_val = 1 if foot == "Right" else 0

height = st.number_input("Height (cm)", 140, 220, 180)
weight = st.number_input("Weight (kg)", 40, 120, 75)
age = st.number_input("Age", 10, 50, 20)

# --- PRZELICZENIE NA SKALĘ EA SPORTS ---
acc = scale_inverse(t10, 1.50, 2.30)
spr = scale_inverse(t30, 3.60, 4.60)
agi = scale_inverse(t_test, 8.5, 13.0)
bal = scale(balance, 3, 45)
jmp = scale(cmj, 15, 75)
sta = scale(coop, 1800, 3500)
strg = scale(grip, 25, 70)


# --- PRZYCISK ---
if st.button("Dopasuj pozycję"):

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

    # prawdopodobieństwa
    probs = model.predict_proba(new_player)[0]
    top3_idx = probs.argsort()[-3:][::-1]

    st.subheader("🏆 TOP 3 dopasowania:")

    for idx in top3_idx:
        pos = encoder.inverse_transform([idx])[0]
        st.write(f"**{pos}** — {probs[idx]*100:.2f}%")

    st.bar_chart(pd.DataFrame({
        "Pozycja": [encoder.inverse_transform([i])[0] for i in top3_idx],
        "Prawdopodobieństwo": [probs[i] for i in top3_idx]
    }).set_index("Pozycja"))
