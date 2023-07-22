# Sentiment Analysis of McDonald's Reviews using BERT
Overview
This project aims to conduct an aspect-based sentiment analysis on a dataset of McDonald's store reviews. Our goal is to identify key terms that describe extremely positive and negative ratings. Additionally, we aim to develop a machine learning model that predicts the rating of a review based on the sentiment expressed in the text.
## Requirements

To run this project, you need the following libraries:

- pandas
- numpy
- scikit-learn
- torch
- transformers
- nltk
- textblob

If you have not installed these libraries, you can install them using pip:

pip install requirements.txt

## Data

The data for this project comes directly from https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews and should be stored in the file 'McDonald_s_Reviews.csv' in the data directeory. Each row in the file corresponds to a unique customer review. The 'review' column contains the text of the review, and the 'rating' column contains the star rating given by the customer.

## Steps to Run the Project

1. Clone this repository to your local machine.
2. Navigate to the project directory in your terminal.
3. Run the following command to execute the script:


python model.py

or 

python aspectanalysis.py


## Results

The results of the model's performance (accuracy, precision, recall, F1 score) will be printed to the console. I will train a model and upload it to huggingface as well, and put the performance on this page

## Author

Eric Modesitt

## License

This project is licensed under the MIT License.