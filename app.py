import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Carregar o modelo treinado
model = tf.keras.models.load_model('fer_model.h5')

# Classes de emoções
emotion_dict = {0: 'Raiva', 1: 'Desgosto', 2: 'Medo', 3: 'Felicidade', 
                4: 'Tristeza', 5: 'Surpresa', 6: 'Neutro'}

# Função para prever emoção
def predict_emotion(img):
    # Redimensionar a imagem para 48x48 pixels
    img = img.resize((48, 48))
    img = img.convert('L')  # Converter para escala de cinza
    img = np.array(img) / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Adicionar batch dimension
    img = np.expand_dims(img, axis=-1)  # Adicionar channel dimension
    
    # Fazer a previsão
    predictions = model.predict(img)
    emotion = np.argmax(predictions)
    return emotion_dict[emotion], predictions

# Configuração da interface do Streamlit
st.title("Reconhecimento de Emoções Faciais")
st.write("Faça upload de uma imagem facial para prever a emoção")

# Upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar a imagem carregada
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem Carregada', use_column_width=True)
    
    # Fazer a previsão da emoção
    emotion, predictions = predict_emotion(image)
    st.write(f"**Emoção Predominante:** {emotion}")
    
    # Mostrar probabilidades de todas as emoções
    st.write("**Probabilidades de cada emoção:**")
    for i, (label, prob) in enumerate(zip(emotion_dict.values(), predictions[0])):
        st.write(f"{label}: {prob*100:.2f}%")