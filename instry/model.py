import dill as pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np

class INSTRYModel:
    def __init__(self, df):
        self.df = df.copy()  # Create a copy to avoid setting values on a slice
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(df['skills'])
        self.designation_vectorizer = TfidfVectorizer()
        self.designation_matrix = self.designation_vectorizer.fit_transform(df['designation'])

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def predict(self, text: str):
        text = self.preprocess_text(text)

        # Step 1: Check if the text has a designation name
        designation_scores = cosine_similarity(self.designation_vectorizer.transform([text]), self.designation_matrix)
        self.df.loc[:, 'designation_score'] = designation_scores[0]
        designation_matches = self.df[self.df['designation_score'] > 0]

        best_matched = None
        alt_matched = None

        if not designation_matches.empty:
            # Step 2: If found more than one designation, generate a confidence score for each and sort
            designation_matches = designation_matches.sort_values(by='designation_score', ascending=False)
            top_designation = designation_matches.iloc[0]
            print(top_designation)

            # Step 3: Predict the designation name and sort by similarity score
            user_tfidf = self.vectorizer.transform([text])
            similarity_scores = cosine_similarity(user_tfidf, self.tfidf_matrix)
            self.df.loc[:, 'similarity'] = similarity_scores[0]
            best_match = self.df.sort_values(by='similarity', ascending=False).iloc[0]
            print("------------")
            print(best_match)

            # Step 4: Compare results from Step 2 and Step 3
            if best_match['designation'] == top_designation['designation']:
                best_matched = best_match
                alt_matched = None
            else:
                best_matched = best_match
                alt_matched = top_designation
        else:
            # No designation match found, use only skill similarity
            user_tfidf = self.vectorizer.transform([text])
            similarity_scores = cosine_similarity(user_tfidf, self.tfidf_matrix)
            self.df.loc[:, 'similarity'] = similarity_scores[0]
            best_matched = self.df.sort_values(by='similarity', ascending=False).iloc[0]
            alt_matched = None

        def to_dict(row):
            result = {col: (row[col].item() if isinstance(row[col], (int, float, complex)) else row[col]) for col in row.index}
            result['designation_score'] = row.get('designation_score', 0.0)
            result['similarity'] = row.get('similarity', 0.0)
            return result

        return {
            'best_matched': to_dict(best_matched),
            'alt_matched': to_dict(alt_matched) if alt_matched is not None else None
        }

def load_model(model_path: str) -> INSTRYModel:
    from .model import INSTRYModel  # Ensure the model class is imported correctly
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
