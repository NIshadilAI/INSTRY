import sys
import pandas as pd
from src.model import INSTRYModel
from src.data_loader import load_data

def main(text: str):
    # Load the data
    data = load_data('data/industry_data.csv')
    
    # Initialize and train the model
    model = INSTRYModel(data)
    
    # Predict the industry
    sector, industry_group, industry, sub_industry = model.predict(text)
    print(f"Sector: {sector}, Industry Group: {industry_group}, Industry: {industry}, Sub-Industry: {sub_industry}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py 'your text here'")
        sys.exit(1)
    
    main(sys.argv[1])
