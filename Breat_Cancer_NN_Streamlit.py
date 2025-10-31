# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import streamlit as st
import pickle


model = load_model("Breast_Cancer_Neural_Network.keras")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    
st.title("Neural Network for Breast Cancer Prediction")

col1, col2, col3, col4 = st.columns(4)


with col1:
  radius_mean = st.text_input('Radius Mean')
with col2:
  texture_mean = st.text_input('Texture Mean')
with col3:
  perimeter_mean = st.text_input('Perimeter_mean')
with col4:
  area_mean = st.text_input('Area mean')
with col1:
  smoothness_mean = st.text_input('Smoothness Mean')
with col2:
  compactness_mean = st.text_input('Compactness Mean')
with col3:
  concavity_mean = st.text_input('Concavity Mean')
with col4:
  concave_points_mean = st.text_input('Concave Points Mean')
with col1:
  symmetry_mean = st.text_input('Symmetry Mean')
with col2:
  fractal_dimension_mean = st.text_input('Fractional Dimension Mean')
with col3:
  radius_se = st.text_input('Radius SE')
with col4:
  texture_se = st.text_input('Texture SE')
with col1:
  perimeter_se = st.text_input('Perimeter SE')
with col2:
  area_se = st.text_input('Area SE')
with col3:
  smoothness_se = st.text_input('Smoothness SE')
with col4:
  compactness_se = st.text_input('Compactness SE')
with col1:
  concavity_se = st.text_input('Concavity SE')
with col2:
  concave_points_se = st.text_input('Concavity Points SE')
with col3:
  symmetry_se = st.text_input('Symmetry SE')
with col4:
  fractal_dimension_se = st.text_input('Fractional Dimension SE')
with col1:
  radius_worst = st.text_input('Radius Worst')
with col2:
  texture_worst = st.text_input('Texture Worst')
with col3:
  perimeter_worst = st.text_input('Perimeter Worst')
with col4:
  area_worst = st.text_input('Area Worst')
with col1:
  smoothness_worst = st.text_input('Smoothness Worst')
with col2:
  compactness_worst = st.text_input('Compactness Worst')
with col3:
  concavity_worst = st.text_input('Concavity Worst')
with col4:
  concave_points_worst = st.text_input('Concave Points Worst')
with col1:
  symmetry_worst = st.text_input('Symmetry Worst')
with col2:
  fractal_dimension_worst = st.text_input('Fractional Dimension Worst')

user_input = (radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst)

diag = ''

if st.button("Test Results"):
    user_input_np = np.asarray(user_input)
    user_input_std_reshaped = user_input_np.reshape(1, -1)
    user_input_std = scaler.transform(user_input_std_reshaped)
    prediction = model.predict(user_input_std)
    
    prediction_label = [np.argmax(prediction)]
    
    if prediction_label[0] == 0:
      diag = "Tumor is Malignant"
    else:
      diag = "Tumor is Benign"
      
    st.success(diag)