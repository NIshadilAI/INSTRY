# INSTRY - Industry Name Identification Model

INSTRY is an AI model designed to identify the industry name from a given text using a dataset of sector, industry group, industry, sub-industry, and industry skills.

## Installation

```bash
pip install INSTRY
```

## Usage

```python
from instry import load_model

model = load_model('path_to_pretrained_model.pkl')
result = model.predict('some text describing skills and job')

print(result)
```