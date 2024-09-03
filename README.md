# 🌱 Crop Recommendation System

This repository contains the implementation of a machine learning model to recommend suitable crops for cultivation based on various environmental factors. The model is trained on the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) available on Kaggle.

## 📊 Dataset

The dataset includes features such as **Nitrogen (N)**, **Phosphorus (P)**, **Potassium (K)**, **Temperature**, **Humidity**, **pH**, and **Rainfall**. It contains data on various crops and their growing conditions, which helps predict the most suitable crop for a given set of environmental conditions.

- **Source:** [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

## 🚀 Project Overview

### Objective

The goal of this project is to develop a **machine learning model** that can accurately recommend a crop for cultivation based on soil and climatic conditions, thus helping farmers in making informed decisions.

### Model

The model uses a **Supervised Learning Algorithm** to predict the crop label. The model is trained on features such as soil nutrients and environmental factors and outputs the best-suited crop. The following steps are involved in the project:

1. **Data Preprocessing:** Cleaning and normalizing the dataset.
2. **Feature Engineering:** Selecting the most relevant features for training.
3. **Model Selection and Training:** Training multiple models and selecting the best-performing one.
4. **Evaluation:** Using metrics like accuracy to evaluate the performance of the model.
5. **Deployment:** The model is deployed using a Flask-based web application.

## ⚙️ Features

- **Input:** Nitrogen, Phosphorus, Potassium levels, Temperature, Humidity, pH level, and Rainfall.
- **Output:** Recommended crop based on the given inputs.
- **User Interface:** The model is deployed with a simple and user-friendly interface that allows users to input their parameters and get crop recommendations.

## 💻 Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Theojims/Crop_Recommendation_System.git
   cd Crop_Recommendation_System
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Open your browser and go to:

   ```bash
   http://localhost:5000/
   ```

## 📈 Results

The model achieves an accuracy of **0.98%**  on the test dataset, and the recommended crops align well with the environmental conditions provided. Further improvements can be made by experimenting with different algorithms and tuning hyperparameters.

## 🔧 Technologies Used

- **Python**
- **Flask** (for web deployment)
- **Machine Learning Libraries:** scikit-learn, pandas, numpy, etc.
- **Data Visualization:** matplotlib, seaborn

## 🎯 Future Work

- Enhance the user interface for better interactivity.
- Incorporate real-time environmental data through APIs.
- Improve the accuracy of the model with advanced algorithms or deep learning.