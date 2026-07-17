import streamlit as st 
import tensorflow as tf
import joblib

# Load model
model = tf.keras.models.load_model("./model/model4_0_758.keras")

model.summary()