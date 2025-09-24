# Flight Price Prediction: An End-to-End Machine Learning Project using AWS Sagemaker

The "Flight Prices Prediction using AWS-Sagemaker" project aims to predict flight prices based on historical data using machine learning techniques. The project uses AWS Sagemaker for model training and deployment, and Streamlit for creating an interactive user interface. The XGBoost model is trained to provide accurate predictions, and Streamlit is used to create an interactive user interface for predictions. The primary goal is to provide users with accurate flight price predictions based on various input features such as airline, source, destination, duration, departure date, and arrival date, etc.

## Features:

1. `Data Storage` : The dataset used for training and prediction is stored in an AWS S3 bucket. Ensure your data is uploaded to the S3 bucket and accessible for Sagemaker.
2. `Feature Engineering` : I used the feature_engine library for feature engineering. This step involves transforming raw data into meaningful features that can be used for model training.
3. `Model Training` : I used XGBoost, a powerful gradient boosting framework, for training our model. The training process is carried out on AWS Sagemaker.
4. `Prediction Interface` : Streamlit is used to create a user-friendly interface for predictions. Users can input data, which is then preprocessed and passed to the trained model for price prediction and displays the predicted price to the interface.
5. `Integration with AWS services using boto3 SDK` : The boto3 SDK is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python. It allows Python to integrate and interact with AWS services seamlessly.

## Results:

The XGBoost model achieved an R2 score of 0.502 on the test data, indicating that the model explains about 50.2% of the variance in flight prices.

## Conclusion:

This project demonstrates a comprehensive machine learning workflow using AWS Sagemaker, from data storage and feature engineering to model training and deployment. The interactive user interface created with Streamlit allows users to input data and get real-time flight price predictions, making the model accessible and user-friendly.

## Live Demo: 

`Click Here` :- https://flight-price-prediction-using-ml-e82kpydwmj7mdmbtvywkat.streamlit.app/#flight-prices-prediction-using-aws-sagemaker

## References:

- AWS Sagemaker Documentation
- XGBoost Documentation
- Streamlit Documentation
- boto3 Documentation
- feature_engine Documentation
