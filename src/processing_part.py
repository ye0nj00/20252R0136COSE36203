# Preprocessing
punct = set(string.punctuation)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in punct]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df['processed_final'] = df['Reviews'].astype(str).apply(preprocess)

# Preprocessed dataset
df_preprocessed = df.copy()
df_preprocessed