# INSTRY - Industry Name Identification Model

INSTRY is an AI model designed to identify the industry name from a given text using a dataset of sector, industry group, industry, sub-industry, and industry skills.

## Installation

```bash
pip install INSTRY
```

## Usage

```python
from instry import load_model
import json

# Make sure to replace 'instry_model.pkl' with the actual path to your model file
model = load_model('instry_model.pkl')
result = model.predict("some text describing skills and job")

# Serialize the result to make it JSON compatible
serialized_result = result  # Result is already serialized by the model

# Print the serialized result as a JSON string
print(json.dumps(serialized_result, indent=4))
```