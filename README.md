# INSTRY - Industry Name Identification Model

INSTRY is an AI model designed to identify the industry name from a given text using a dataset of sector, industry group, industry, sub-industry, and industry skills.

## Dataset

The dataset contains the following columns:
- sector
- industry_group
- industry
- sub_industry
- industry_skills

## Installation

1. Create a virtual environment and activate it.
2. Install the necessary packages.
3. Download NLTK data.

```bash
python -m venv instry-env
source instry-env/bin/activate
pip install -r requirements.txt
python -m nltk.downloader all
