import streamlit as st
import pickle
import numpy as np

# Load your trained model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction")

# Input fields (change as needed)
Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Age = st.number_input("Age", min_value=0.0, max_value=100.0, step=0.1)
Fare = st.number_input("Fare", min_value=0.0, step=0.1)
FamilySize = st.number_input("Family Size", min_value=0, step=1)
Sex_enc = st.selectbox("Sex (0=Male, 1=Female)", [0, 1])
Embarked_C = st.selectbox("Embarked at C (0 or 1)", [0, 1])
Embarked_Q = st.selectbox("Embarked at Q (0 or 1)", [0, 1])
Embarked_S = st.selectbox("Embarked at S (0 or 1)", [0, 1])
SibSp = st.number_input("Number of siblings/spouses aboard (SibSp)", min_value=0, step=1)

if st.button("Predict Survival"):
    # Prepare input array with exactly 9 features
    input_array = np.array([[Pclass, Age, Fare, FamilySize, Sex_enc,
                             Embarked_C, Embarked_Q, Embarked_S, SibSp]])

    # Debug prints (optional)
    st.write("Input shape:", input_array.shape)
    st.write("Input data:", input_array)

    # Prediction
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did Not Survive")
