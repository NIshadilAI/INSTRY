from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class INSTRYModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(df['skills'].apply(lambda x: ' '.join(x)))

    def predict(self, text: str) -> pd.Series:
        user_tfidf = self.vectorizer.transform([text])
        similarity_scores = cosine_similarity(user_tfidf, self.tfidf_matrix)
        self.df['similarity'] = similarity_scores[0]
        best_match = self.df.sort_values(by='similarity', ascending=False).iloc[0]
        return best_match
