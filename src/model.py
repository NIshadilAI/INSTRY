from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class INSTRYModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(data['industry_skills'])

    def predict(self, text: str):
        text_tfidf = self.vectorizer.transform([text])
        similarity_scores = cosine_similarity(text_tfidf, self.tfidf_matrix).flatten()
        best_match_idx = similarity_scores.argmax()
        best_match = self.data.iloc[best_match_idx]
        return best_match['sector'], best_match['industry_group'], best_match['industry'], best_match['sub_industry']
