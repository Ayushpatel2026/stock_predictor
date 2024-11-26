# Stock Price Predictor

This project implements a machine learning model to predict stock prices using a Multilayer Perceptron (MLP) and serves predictions through a Flask API. 
The project involves data exploration, feature engineering, model training, and deployment through a web interface.

## Table of Contents
- [Overview](#overview)
- [Data Exploration](#data-exploration)
- [Data Cleaning/Encoding](#data-cleaning/encoding)
- [Feature Engineering](#feature-engineering)
- [Data Scaling](#data-scaling)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Flask API](#flask-api)
- [Conclusion](#conclusion)

---

## Overview
The goal of this project is to predict stock closing prices based on several input features, such as the high, low, open prices, trading volume, and date-related information. 
A deep learning model was built and trained on historical stock data, utilizing a Multilayer Perceptron (MLP) with three hidden layers. 
The technologies/libraries used for this project include Python, PyTorch, NumPy, pandas, matplotlib, scikit-learn and Flask API. 

## Data Exploration
Before training the model, an in-depth data analysis phase was conducted:
- **Heatmaps and Correlation Plots** were created to visualize relationships between features such as high, low, open prices, and volume. This helped in understanding feature correlations.
- Exploratory Data Analysis (EDA) showed that certain features like the date and stock prices were highly correlated, which will help guide further feature engineering.

## Data Cleaning/Encoding 
The input data included dates and company names, which required proper encoding:
- **Date Encoding**: The date column was broken down into individual components—`year`, `month`, and `day`—to capture temporal patterns.
- **Company Name Encoding**: Categorical values like company names were encoded using **Label Encoding** to convert them into numerical form while retaining their categorical nature.

## Feature Engineering
The core model does not contain much feature engineering, but I am working on a more complex model that will include features such as rolling average, volatility, RSI etc. 

## Data Scaling
To ensure the model interprets each feature fairly, **MinMaxScaler** was used to scale the data:
- **Stock Prices** (high, low, open, close) and **Volume** vary greatly in magnitude compared to categorical features like `name_encoded`. To account for this, the data was scaled so that all features fell within a similar range.
- Scaling is particularly important because categorical variables like `name_encoded` (company names) might otherwise introduce bias due to their numerical representation.

## Model Architecture
The Multilayer Perceptron (MLP) used in this project has the following architecture:
- **Input Size**: Number of features
- **Hidden Layers**: 
  - 3 hidden layers with **128**, **64**, and **64** neurons, respectively, with a total of **13633** parameters. 
  - **Dropout layers** were included to prevent overfitting, with a probability of 0.2. 
  - **Tanh activation function** was used between layers to introduce non-linearity.

## Model Training
**Core Model**
The core model was trained on historical stock data with the following settings:
- **Number of Epochs**: 10,000
- **Batch Size**: 256
- **Loss Function**: Mean Squared Error (MSE) for regression tasks.
- **Optimizer**: Adam optimizer with starting learning rate of 0.0001, and decay of 0.8 every 2000 epochs for better convergence
        This algorithm keeps track of exponential decaying average of past gradients and past squared gradients, helping to smooth out updates and scale the learning rate for each parameter, respectively. 

**Training results:**
- **Training Loss**: 0.0002484
- **Validation Loss**: 0.0000107
- **Test Loss**: 0.0000129

The results suggest that the model has generalized well, as the validation and test losses are close in value.
However, the higher training loss indicated potential batch size issues or model fluctuations during training and I may experiment with a Batch Normalization Layer to mitigate this problem. 

**Feature engineered model**
  COMING SOON!

## Evaluation Metrics
**Core Model**
The model's performance was evaluated using:
- **Mean Absolute Percentage Error (MAPE)**: 6.73% — This means that, on average, the model's predictions are within 6.73% of the actual stock prices, which is a reasonably good performance.
- **Root Mean Squared Error (RMSE)**: 7.28 — This gives a sense of how much the model's predictions deviate from the actual values, measured in the same units as the stock prices. On average, the model's prediction is +/- $7.28
**Feature engineered model**
  COMING SOON!

## Flask API
The trained model was deployed as a **Flask API** to allow users to input stock data (including date, volume, company name, high, low, and open prices) and receive predictions for the closing price. 

- The API accepts the input values, processes them (including encoding company names and scaling the input), makes a prediction using the MLP model, and returns the predicted closing price. 
- The front-end interface includes an HTML form that collects user inputs and displays the predicted closing price.

## Usage
1. To run locally either run from the IDE or use the following command:
```bash
python app.py
```
2. Open your web browser and visit `http://localhost:3000`
3. Enter the stock information and click on the 'predict' button

## Conclusion
This project successfully predicts stock prices based on historical data using a deep learning model. 
The deployment of the Flask API allows users to interact with the model easily and provides accurate stock price predictions with relatively low error rates. 

---
