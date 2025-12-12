import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =============================
#   ŁADOWANIE MODELU Z CACHE
# =============================
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, encoder

model, encoder = load_model()


# =============================
#   FUNKCJE PRZELICZAJĄCE
# =============================
def scale(x, min_val, max_val):
    return int(1 + 98 * (x - min_val) / (max_val - min_val))

def scale_inverse(x, min_val, max_val):
    return int(1 + 98 * (max_val - x) / (max_val - min_val))


# =============================
#   INTERFEJS
# =============================
st.title("⚽ Predykcja optymalnej pozycji piłkarskiej")
st.write("Wprowadź wyniki testów, a aplikacja pokaże TOP 3 najlepiej dopasowane pozycje boiskowe.")


# =====================================================
#   FORMULARZ (zapobiega restartom przy każdym suwaku)
# =====================================================
with st.form("input_form"):

    st.header("Testy sprawnościowe")

    t10 = st.number_input("Czas biegu na 10 m (s)", 1.5, 4.0, 2.0)
    t30 = st.number_input("Czas biegu na 30 m (s)", 3.3, 7.0, 4.2)
    t_test = st.number_input("T-test agility (s)", 8.0, 20.0, 11.0)
    balance = st.number_input("Stanie na 1 nodze (s)", 1, 60, 20)
    cmj = st.number_input("Wyskok pionowy CMJ (cm)", 10, 100, 40)
    coop = st.number_input("Test Coopera – dystans (m)", 1000, 4000, 2500)
    squat = st.number_input("Przysiad ze sztangą 1RM (kg)", 30, 200, 80)

    st.header("Dane antropometryczne")

    foot = st.radio("Preferowana noga", ["Right", "Left"])
    foot_val = 1 if foot == "Right" else 0

    height = st.number_input("Wysokość (min 140, max 220)", 140, 220, 180)
    weight = st.number_input("Waga (kg)", 40, 120, 75)
    age = st.number_input("Wiek", 10, 50, 20)

    # TRIK: blokada ENTER
    st.form_submit_button("...", type="secondary", disabled=True)

    # Właściwy przycisk — tylko on wywołuje obliczenia
    submitted = st.form_submit_button("Oblicz pozycję", type="primary")


# =============================
#   PRZELICZENIA I PREDYKCJA
# =============================
if submitted:

    acc = scale_inverse(t10, 1.50, 2.30)
    spr = scale_inverse(t30, 3.60, 5.00)
    agi = scale_inverse(t_test, 8.5, 13.0)
    bal = scale(balance, 3, 45)
    jmp = scale(cmj, 25, 75)
    sta = scale(coop, 1800, 3600)
    strg = scale(squat, 40, 180)

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

    probs = model.predict_proba(new_player)[0]
    top3_idx = probs.argsort()[-3:][::-1]

    st.subheader("🏆 TOP 3 dopasowania:")

    for idx in top3_idx:
        pos = encoder.inverse_transform([idx])[0]
        st.write(f"**{pos}** — {probs[idx]*100:.2f}%")

    df_chart = pd.DataFrame({
        "Pozycja": [encoder.inverse_transform([i])[0] for i in top3_idx],
        "Prawdopodobieństwo": [probs[i] for i in top3_idx]
    }).set_index("Pozycja")

    st.bar_chart(df_chart)
