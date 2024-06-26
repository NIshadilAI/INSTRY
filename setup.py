from setuptools import setup, find_packages

setup(
    name='INSTRY',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'nltk'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='An AI model to identify industry names from given text',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NIshadilAI/INSTRY',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
