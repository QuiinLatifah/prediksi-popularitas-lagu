import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vectorizer
model = pickle.load(open('predict_lirik.pkl', 'rb'))
vectorizer = pickle.load(open('tf_idf_vectorizer.pkl', 'rb'))

# Judul aplikasi
st.markdown("<h1 style='text-align: center; color: darkblue;'>Prediksi Popularitas Lagu Berdasarkan Lirik</h1>", unsafe_allow_html=True)

# Input untuk lirik
lyric = st.text_area("Tulis Lirik Lagu di sini", height=300, placeholder="Contoh: Untungnya, bumi masih berputar...")

lirik_predict = ''

# Tombol prediksi
if st.button('Prediksi Popularitas'):
    if not lyric:
        st.warning("Silakan masukkan lirik lagu terlebih dahulu.")
    else:
        # Transformasi lirik yang diinput menggunakan vectorizer yang telah diload
        lyric_transformed = vectorizer.transform([lyric])

        # Prediksi menggunakan model yang telah diload
        predict_lirik = model.predict(lyric_transformed)

        # Tampilkan hasil prediksi
        if predict_lirik == 'sangat populer':
            st.success("Prediksi popularitas lirik lagu ini adalah: **Sangat Populer** 🎉")
        elif predict_lirik == 'lumayan populer':
            st.info("Prediksi popularitas lirik lagu ini adalah: **Lumayan Populer** 😊")
        elif predict_lirik == 'kurang populer':
            st.warning("Prediksi popularitas lirik lagu ini adalah: **Kurang Populer** 😕")