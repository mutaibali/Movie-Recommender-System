# Movie Recommendation System 🎥

This project is a **Content-Based Movie Recommendation System** that suggests movies based on user preferences by analyzing movie metadata such as genres, keywords, cast, crew, and descriptions. The system uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to compute similarity between movies and generate recommendations.

---

## 📌 Project Overview
- **Purpose**: To recommend movies similar to a given movie based on its metadata.
- **Core Features**:
  - Clean and preprocess movie metadata.
  - Generate meaningful tags from genres, keywords, cast, crew, and movie descriptions.
  - Use **CountVectorizer** and **Cosine Similarity** to compute similarity scores.
  - Provide a list of top 5 similar movies for any given movie.

---

## 🛠️ Skills Demonstrated
- **Data Preprocessing**:
  - Cleaning missing and duplicate data.
  - Extracting useful information from complex JSON-like columns.
- **Text Processing**:
  - Tokenization, stemming, and removing stop words for feature extraction.
- **Machine Learning**:
  - Vectorizing textual data using **CountVectorizer**.
  - Calculating similarity using **Cosine Similarity**.
- **Python Programming**:
  - Pandas, NumPy, Scikit-learn, and NLTK.

---

## 📂 Datasets
- **TMDB 5000 Movies Dataset**:
  - `tmdb_5000_movies.csv`: Contains movie metadata such as genres, keywords, and overview.
  - `tmdb_5000_credits.csv`: Contains information about cast and crew.

---

## 🔍 Workflow and Key Steps

### 1. **Data Loading**
   - **Purpose**: Load the datasets and merge them to create a unified dataset for analysis.
   - **Code**:
     ```python
     movies = pd.read_csv('tmdb_5000_movies.csv')
     credits = pd.read_csv('tmdb_5000_credits.csv')
     movies = movies.merge(credits, on='title')
     ```

### 2. **Data Cleaning**
   - **Purpose**: Clean missing and duplicate data to ensure consistency and accuracy.
   - **Steps**:
     - Remove null values.
     - Drop duplicate entries.
     - Select only the relevant columns: `movie_id`, `title`, `genres`, `keywords`, `overview`, `cast`, and `crew`.
   - **Code**:
     ```python
     movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]
     movies.dropna(inplace=True)
     ```

### 3. **Feature Extraction**
   - **Purpose**: Extract meaningful features from the `genres`, `keywords`, `cast`, and `crew` columns.
   - **Steps**:
     - Convert JSON-like strings in these columns into Python lists.
     - Extract the top 3 actors from the `cast` column.
     - Identify and extract the director from the `crew` column.
   - **Code**:
     ```python
     def convert(obj):
         L = []
         for i in ast.literal_eval(obj):
             L.append(i['name'])
         return L

     movies['genres'] = movies['genres'].apply(convert)
     movies['keywords'] = movies['keywords'].apply(convert)
     ```

### 4. **Tag Creation**
   - **Purpose**: Combine important features into a single column (`tags`) for analysis.
   - **Steps**:
     - Concatenate `overview`, `genres`, `keywords`, `cast`, and `crew`.
     - Normalize the text by converting it to lowercase and removing spaces.
   - **Code**:
     ```python
     movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
     movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())
     ```

### 5. **Text Vectorization**
   - **Purpose**: Convert the `tags` column into numerical vectors for analysis.
   - **Steps**:
     - Use `CountVectorizer` to create vectors with a maximum of 5000 features.
     - Remove common stop words.
   - **Code**:
     ```python
     from sklearn.feature_extraction.text import CountVectorizer
     cv = CountVectorizer(max_features=5000, stop_words='english')
     vectors = cv.fit_transform(movies['tags']).toarray()
     ```

### 6. **Cosine Similarity**
   - **Purpose**: Measure the similarity between movies based on their vector representations.
   - **Code**:
     ```python
     from sklearn.metrics.pairwise import cosine_similarity
     similarity = cosine_similarity(vectors)
     ```

### 7. **Recommendation Function**
   - **Purpose**: Recommend the top 5 most similar movies for a given movie title.
   - **Code**:
     ```python
     def recommend(movie):
         movie_index = new_df[new_df['title'] == movie].index[0]
         distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[1:6]
         for i in distances:
             print(new_df.iloc[i[0]].title)
     ```

---

## 🚀 How to Use
1. **Set Up**:
   - Download the `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` datasets.
   - Install required Python libraries: `pandas`, `numpy`, `scikit-learn`, `nltk`.

2. **Run the Script**:
   - Execute the Python script to load and process the data.
   - Use the `recommend()` function to get movie recommendations:
     ```python
     recommend('Avatar')
     ```

3. **Example Output**:
   For the movie "Avatar," the system might recommend:

  - Guardians of the Galaxy
  - Avengers: Age of Ultron
  - Interstellar
  - Star Trek Beyond
  - Thor




---

## 🔮 Future Enhancements
- Use **TF-IDF Vectorizer** for better text representation.
- Integrate **Deep Learning** models like Word2Vec or BERT for more robust recommendations.
- Add user ratings to build a **Hybrid Recommendation System**.
- Create a web interface using Flask or Streamlit for easy interaction.

---

## 🤝 Connect
Feel free to reach out for questions or suggestions:
- **Email**: [syedmutaib0599@gmail.com](mailto:syedmutaib0599@gmail.com)
- **LinkedIn**: [Syed Mutaib Ali](https://linkedin.com/in/syedmutaibali)

---

This updated README ensures consistency across all sections while maintaining clarity and detail. Let me know if you’d like further refinements! 😊
