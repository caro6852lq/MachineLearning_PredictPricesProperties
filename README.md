# MachineLearning_PredictPricesProperties
This repository contein the midterm project for course of Machine Learning in Data Talks Club

-----------------------------------------------------------------------------------------------

🏙️ Property Price Prediction in Buenos Aires (CABA)


📌 Project Description

The goal of this project is to develop a Machine Learning model capable of predicting the sale price of real estate properties in Buenos Aires (CABA) based on features such as location, surface area, number of rooms, bathrooms, property type, and age.

This model aims to:

- Estimate property market prices automatically.

- Identify underpriced or overpriced listings.

- Explore housing market trends across different neighborhoods in the city.

---- REVISAR!

📂 Repository Structure
├── data/
│   ├── properties_caba.csv         # Dataset with property listings
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb    # Feature processing and selection
│   ├── 03_model_training.ipynb         # Model training
│   ├── 04_evaluation.ipynb             # Model evaluation
│
├── models/
│   ├── model_final.pkl                 # Trained model
│
├── requirements.txt                    # Project dependencies
├── README.md                           # This file

⚙️ How to Run the Project

Clone the repository

git clone https://github.com/your-username/property-price-prediction-caba.git
cd property-price-prediction-caba


Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows


Install dependencies

pip install -r requirements.txt


Run the notebooks

Open Jupyter Notebook or VSCode and run the notebooks in /notebooks in order, starting with the exploratory analysis.

🧠 Models and Metrics

Several regression models were tested, including:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

The main evaluation metric used was the Root Mean Squared Error (RMSE), which measures the average difference between predicted and actual prices.

🚀 Next Steps

Add geographic features (latitude/longitude) for better accuracy.

Deploy the model as a web service using Flask or FastAPI.

Build an interactive dashboard to visualize predictions.

👩‍💻 Author

Carolina Vergara
Midterm Project — Machine Learning Zoomcamp (DataTalks.Club)
📅 October 2025
