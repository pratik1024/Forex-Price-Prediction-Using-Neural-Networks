# Forex-Price-Prediction-Using-Neural-Networks

A Python-based project to predict forex prices using a single-neuron neural network. The project demonstrates end-to-end implementation of a neural network from scratch, including data preprocessing, model building, and error evaluation. It focuses on leveraging Python libraries such as NumPy, Pandas, and scikit-learn for financial time series data analysis.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)


---

## **Introduction**
Forex trading involves predicting currency price fluctuations. This project implements a simple neural network model to predict forex prices based on historical data. The model is built from scratch, demonstrating a clear understanding of machine learning fundamentals and time series prediction.

---

## **Features**
- Built a single-neuron neural network from scratch using Python.
- Normalized input data for better model performance.
- Utilized a stride-tricks approach for time series data preprocessing.
- Calculated Root Mean Square Error (RMSE) to evaluate model accuracy.
- Predicts forex prices with minimal error (RMSE = 0.00039).

---

## **Technologies Used**
- **Programming Language:** Python  
- **Libraries:** 
  - NumPy (Matrix operations and feature engineering)
  - Pandas (Data manipulation)
  - scikit-learn (Data preprocessing and evaluation)

---

## **Dataset**
- **Source:** Forex price data (add source link if applicable).  
- The dataset includes historical forex prices over a specified period.  
- Preprocessed for feature engineering and used for training and testing the model.

---

## **Project Workflow**
1. **Data Preprocessing**
   - Load and normalize the dataset.
   - Use `stride_tricks` for transforming time series data into a supervised learning format.

2. **Model Implementation**
   - Built a single-neuron neural network from scratch, including:
     - Weight initialization
     - Activation function
     - Gradient-based weight updates.

3. **Evaluation**
   - Predicted forex prices using the trained model.
   - Calculated RMSE for model performance evaluation.

4. **Visualization**
   - Plotted the predicted vs actual values for visual assessment of the model.

---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/forex-price-prediction.git
   cd forex-price-prediction
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```


3. Run the project:
   ```bash
   final_model.py
   ```

---

## **Usage**
- **Main Script:** `main.py`  
  Run this script to preprocess data, train the model, and predict forex prices.  
- Customize the parameters in the script, such as:
  - Input file path
  - Learning rate
  - Number of epochs.

---

## **Results**
- RMSE achieved: **0.00043**  
- The predicted forex prices closely match actual prices, as seen in the visualization:
  
  ![image](https://github.com/user-attachments/assets/bb0a1b39-a46e-4a9d-9712-133dc1026f0c)


---

## **Future Enhancements**
- Add multi-layer neural networks for better predictions.
- Implement advanced algorithms such as LSTMs for improved time series forecasting.
- Integrate live forex data using APIs for real-time predictions.

---
