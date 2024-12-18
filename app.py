import pickle
import streamlit as st
import pandas as pd  # Import pandas to load your dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load model and vectorizer with error handling
def load_model_and_vectorizer():
    try:
        model = pickle.load(open('lyrics_predict.pkl', 'rb'))
        vectorizer = pickle.load(open('Vectorizer.pkl', 'rb'))
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

if model is None or vectorizer is None:
    st.stop()  # Stop the app if model or vectorizer fails to load

# Load dataset
data = pd.read_csv('indo_17-24_clean.csv')  # Ganti dengan path dataset Anda

# Sidebar untuk memilih sentimen
st.sidebar.title("Pilih Sentimen")
sentimen_options = ["sangat populer", "lumayan populer", "kurang populer"]
selected_sentiment = st.sidebar.radio("Lihat WordCloud untuk Sentimen:", sentimen_options)

# Tentukan colormap sesuai sentimen
colormap_dict = {
    "sangat populer": "viridis",
    "lumayan populer": "plasma",
    "kurang populer": "rocket"
}
selected_colormap = colormap_dict[selected_sentiment]

# Filter dataFrame sesuai sentimen yang dipilih
data_filtered = data[data['sentiment'] == selected_sentiment]

# Gabungkan semua teks dalam kolom 'lyrics_clean' untuk sentimen yang dipilih
all_words_filtered = ' '.join(data_filtered['lyrics_clean'])

# Generate WordCloud
wordcloud = WordCloud(
    background_color='white', 
    width=800, height=500, 
    random_state=21, 
    max_font_size=130, 
    colormap=selected_colormap  # Colormap sesuai sentimen
).generate(all_words_filtered)

# Tampilkan WordCloud di sidebar
st.sidebar.subheader(f"WordCloud Sentimen '{selected_sentiment.capitalize()}'")
st.sidebar.image(wordcloud.to_array(), use_column_width=True)

# Hitung 10 kata paling sering muncul untuk sentimen yang dipilih
word_freq_filtered = Counter(all_words_filtered.split())
most_common_words_filtered = word_freq_filtered.most_common(10)

# Tampilkan barplot di sidebar
st.sidebar.subheader("Visualisasi 10 Kata yang Sangat Populer")
words, counts = zip(*most_common_words_filtered)
fig, ax = plt.subplots(figsize=(4, 3))  # Ukuran lebih kecil
sns.barplot(x=list(counts), y=list(words), palette=selected_colormap, ax=ax)
ax.set_title("", fontsize=10)  # Hilangkan judul
ax.set_xlabel("Frekuensi", fontsize=8)
ax.set_ylabel("Kata", fontsize=8)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
st.sidebar.pyplot(fig)

# Judul aplikasi
st.markdown("<h1 style='text-align: center; color: darkblue;'>Prediksi Popularitas Lagu Berdasarkan Lirik</h1>", unsafe_allow_html=True)

# Input untuk lirik
lyric = st.text_area("Tulis Lirik Lagu di sini", height=300, placeholder="Contoh: Untungnya, bumi masih berputar...")

lirik_predict = ''

# Tombol prediksi
if st.button('Prediksi'):
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