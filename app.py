import pandas as pd 
import numpy as np 
import streamlit as st 
import pickle as pk 

model = pk.load(open('D:\Python\Projects\Heart Disease Prediction\models\Model_Analysis.pkl', 'rb'))

data = pd.read_csv('D:\Python\Projects\Heart Disease Prediction\dataset\heart_disease.csv')

st.header('Heart Disease Predictor')

gender = st.selectbox('Chose Gender', data['Gender'].unique())
if( gender == 'Male'):
    gen = 1
else :
    gen = 0 

age = st.number_input("Enter Age : ")
currentSmoker = st.number_input("Is Patient Smoker : ")
cigsPerDay = st.number_input("Enter Cigaretes Per Day  : ")
BPMeds = st.number_input("Is Patient BP Meds : ")
prevalentStroke = st.number_input("Does Patient had a Stroke Before : ")
prevalentHyp = st.number_input("Enter Prevelant Hyp Status : ")
diabetes = st.number_input("Enter Diabetes Status : ")
totChol = st.number_input("Enter Total Cholestrol : ")
sysBP = st.number_input("Enter sysBP : ")
diaBP = st.number_input("Enter diaBP : ")
BMI = st.number_input("Enter BMI : ")
heartRate = st.number_input("Enter Heart Rate : ")
glucose = st.number_input("Enter Glucose : ")

if st.button('Predict') :
    X = np.array([[gen, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose ]])

    output = model.predict(X)
    if output[0] == 0 :
        stn = "Patient is Healthy , No Heart Disease "
    else:
        stn = "Patient May Have a Heart Disease " 
    
    st.markdown(stn) 

