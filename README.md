
#Arabic Text Classification using AraBERT
#Overview

This project is an Arabic text classification system built using AraBERT.
It classifies Arabic sentences into multiple categories using both fine-tuned and non-fine-tuned models.

#Features
Arabic text preprocessing (stopwords removal + normalization)
Two model options:
Fine-tuned AraBERT
Pretrained AraBERT (no fine-tuning)
User-friendly GUI built with Tkinter
Displays:
Evaluation metrics (Accuracy, Precision, Recall, F1-score)
Confusion matrix
Prediction results
Saves results into MySQL database
#Technologies Used
Python
PyTorch
TensorFlow (optional if used)
AraBERT
Tkinter (GUI)
MySQL
#Dataset
~1000 Arabic sentences
Organized into 10 categories (e.g., sports, entertainment, etc.)

Note: If you want the dataset i worked on you can contact me At beghdaouihessine@gmail.com

#Installation
git clone https://github.com/your-username/arabic-text-classification-arabert.git
cd arabic-text-classification-arabert
pip install -r requirements.txt
#How to Run
python main.py
#Database Setup
Create a MySQL database:
CREATE DATABASE text_classification;
Import the schema:
mysql -u root -p text_classification < database/schema.sql
#Results
High accuracy using fine-tuned AraBERT
Better performance compared to non-fine-tuned model
