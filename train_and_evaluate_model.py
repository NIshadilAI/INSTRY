import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dill as pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from instry.preprocessing import load_dataset, preprocess_data
from instry.model import INSTRYModel  # Ensure the model class is imported correctly

# Download NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    return ' '.join(words)

def train_and_save_model(dataset_path: str, model_path: str):
    df = load_dataset(dataset_path)
    df = preprocess_data(df)
    model = INSTRYModel(df)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = [model.predict(text)['best_matched']['sub_industry'] for text in X_test]
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    dataset_path = 'instry_dataset.csv'
    model_path = 'instry_model.pkl'
    
    # Load and preprocess the dataset
    df = load_dataset(dataset_path)
    df = preprocess_data(df)
    
    # Split the data into training and testing sets
    X = df['skills']
    y = df['sub_industry']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model and save it
    train_df = df[df.index.isin(X_train.index)]
    model = INSTRYModel(train_df)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
