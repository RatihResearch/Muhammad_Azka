import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st
import joblib
import nltk
from difflib import SequenceMatcher

# Inisialisasi stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Definisikan stopword
stop_factory = StopWordRemoverFactory()
more_stopword = [
    'kilogram', 'kg', 'gram', 'g', 'gr', 'kecil', 'sedang', 'besar', 'sedikit', 'secukupnya', 'lembar',
    'lbr', 'siung', 'mangkok', 'ruas', 'piring', 'matang', 'btg', 'bh', 'masak', 'potong', 'ptng', 'masak',
    'cincang', 'iris', 'cuci', 'bersih', 'buah', 'buahnya', 'bungkus', 'bks', 'iris', 'serut', 'butir', 'biji', 'stgh',
]

df_stopword = stop_factory.get_stop_words() + more_stopword

def preprocess(s):
    s = re.sub(r'[^\w\s\d\n]', ' ', s)
    s = re.sub(r'\d+', ' ', s)

    hasil = []
    word_token = s.split()  # Menggunakan split() untuk tokenisasi
    unique_words = set()

    for word in word_token:
        word = word.strip().lower()
        if word not in df_stopword:
            word = stemmer.stem(word)
            if word not in unique_words:
                hasil.append(word)
                unique_words.add(word)
    result_sentence = " ".join(hasil).strip()
    print(f"Preprocessed text: {result_sentence}")  # Print hasil preprocessing
    return result_sentence

def stream_matching(input_text):
    matched_nutrients = []
    
    # Keywords for high protein foods
    high_protein_foods = [
        'daging sapi', 'daging kambing', 'daging rusa', 'daging ayam', 'daging bebek', 'ikan', 'seafood', 
        'telur', 'susu', 'kedelai', 'kacang tanah', 'keju', 'tahu', 'tempe', 'udang', 'ikan tuna', 'ikan salmon',
        'ayam potong', 'ayam kampung', 'sarden'
    ]

    # Keywords for carbohydrate foods
    carbohydrate_foods = [
        'beras putih', 'roti gandum', 'sereal sarapan', 'kentang', 'ubi jalar', 'singkong', 'wortel', 
        'quinoa', 'oatmeal', 'sorghum', 'nasi', 'roti tawar', 'nasi merah', 'pasta', 'ubi manis', 'katuk',
        'mie', 'mi instan', 'lontong', 'roti bakar'
    ]

    # Keywords for sodium foods
    sodium_foods = [
        'sosis', 'ham', 'bacon', 'daging asap', 'garam', 'miso', 'sawi asin', 'telur asin', 'ikan asin',
        'kerupuk', 'kornet', 'kaldu instan', 'sarden', 'keju asin', 'saus tiram', 'teri', 'dendeng'
    ]

    # Keywords for fiber foods
    fiber_foods = [
        'apel', 'pir', 'pisang', 'beri', 'jeruk', 'brokoli', 'bayam', 'wortel', 'kacang polong',
        'kacang almond', 'biji chia', 'biji rami', 'kacang hijau', 'kacang merah', 'bubur oat', 'popcorn',
        'cabe rawit', 'kubis', 'kacang panjang', 'ubijalar', 'avocado', 'kiwi', 'sawi hijau', 'tauge', 'labu siam'
    ]

    # Keywords for fat foods
    fat_foods = [
        'alpukat', 'minyak zaitun', 'kacang-kacangan', 'biji-bijian', 'ikan berlemak', 'mentega', 'keju',
        'daging merah', 'minyak kelapa', 'telur bebek', 'kacang kedelai', 'kacang kenari', 'kacang hazelnut',
        'avocado', 'susu kental manis', 'santan', 'mentega kacang', 'lemak babi', 'lemak sapi'
    ]

    # Keywords for potassium foods
    potassium_foods = [
        'pisang', 'jeruk', 'alpukat', 'melon', 'bayam', 'ubi jalar', 'kentang', 'tomat', 'kacang merah',
        'kacang hitam', 'lentil', 'buncis', 'salmon', 'cod', 'tuna', 'kismis', 'kurma', 'terong',
        'paprika', 'semangka', 'kacang hijau', 'labu', 'labu kuning', 'sawi putih', 'kangkung'
    ]

    # Keywords for cholesterol foods
    cholesterol_foods = [
        'daging merah', 'daging unggas', 'telur', 'produk susu', 'hati', 'otak', 'tahung', 'udang', 'kerang', 'tiram',
        'kuning telur', 'keju cheddar', 'usus ayam', 'usus sapi'
    ]

    # Keywords for sugar foods
    sugar_foods = [
        'permen', 'kue', 'biskuit', 'es krim', 'soda', 'jus buah', 'madu', 'gula merah', 'maple syrup', 'stevia',
        'sirup', 'selai', 'brown sugar', 'gula pasir', 'gula jawa', 'sari buah', 'cokelat', 'stroberi',
        'karamel', 'cendol', 'susu kental manis', 'syrup maple'
    ]

    # Perform stream matching for each nutrient category
    for keyword in high_protein_foods:
        if SequenceMatcher(None, input_text, keyword).ratio() > 0.7:
            matched_nutrients.append('Protein')
            break
    
    for keyword in carbohydrate_foods:
        if SequenceMatcher(None, input_text, keyword).ratio() > 0.7:
            matched_nutrients.append('Carbohydrate')
            break
    
    for keyword in sodium_foods:
        if SequenceMatcher(None, input_text, keyword).ratio() > 0.7:
            matched_nutrients.append('Sodium')
            break
    
    for keyword in fiber_foods:
        if SequenceMatcher(None, input_text, keyword).ratio() > 0.7:
            matched_nutrients.append('Fiber')
            break
    
    for keyword in fat_foods:
        if SequenceMatcher(None, input_text, keyword).ratio() > 0.7:
            matched_nutrients.append('Fat')
            break
    
    for keyword in potassium_foods:
        if SequenceMatcher(None, input_text, keyword).ratio() > 0.7:
            matched_nutrients.append('Potassium')
            break
    
    for keyword in cholesterol_foods:
        if SequenceMatcher(None, input_text, keyword).ratio() > 0.7:
            matched_nutrients.append('Cholesterol')
            break
    
    for keyword in sugar_foods:
        if SequenceMatcher(None, input_text, keyword).ratio() > 0.7:
            matched_nutrients.append('Sugar')
            break
    
    return matched_nutrients

# Load the trained models and TF-IDF vectorizer
multi_target_forest = joblib.load('multi_target_forest_model.pkl')
multi_target_nb = joblib.load('multi_target_nb_model.pkl')
multi_target_dt = joblib.load('multi_target_dt_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
Y_columns = joblib.load('Y_columns.pkl')  # Assume Y columns are stored in a separate file

# Download necessary NLTK data
nltk.download('punkt')

# Title
st.title('Nutrition Detection on Food Composition')

# Input text
food_input = st.text_input('Input Komposisi Makanan:')

# Choosing model
model_choice = st.selectbox('Choose a model:', ['Random Forest', 'Naive Bayes', 'Decision Tree'])

# Button to predict
if st.button('Predict'):
    if food_input:
        # Preprocess the input using the preprocessing function
        preprocessed_input = preprocess(food_input)

        # Stream Matching
        matched_nutrients = stream_matching(preprocessed_input)

        if matched_nutrients:
            st.write('Detected Nutrients:', ', '.join(matched_nutrients))
        else:
            # Transform the preprocessed input using the TF-IDF vectorizer
            input_features = tfidf_vectorizer.transform([preprocessed_input])

            # Predict using the chosen model
            if model_choice == 'Random Forest':
                prediction = multi_target_forest.predict(input_features)
            elif model_choice == 'Naive Bayes':
                prediction = multi_target_nb.predict(input_features)
            elif model_choice == 'Decision Tree':
                prediction = multi_target_dt.predict(input_features)

            # Assuming prediction is a multi-label binary array
            nutritions = ['Protein', 'Carbohydrate', 'Sodium', 'Fiber', 'Fat', 'Potassium', 'Cholesterol', 'Sugar']
            detected_nutritions = [nutritions[i] for i in range(len(nutritions)) if prediction[0][i] == 1]

            if detected_nutritions:
                st.write('Detected Nutrition:', ', '.join(detected_nutritions))
            else:
                st.write('No Nutrition Detected.')
    else:
        st.write('Please enter the food description.')
