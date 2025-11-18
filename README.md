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

⚙️ How to Run the Project

Follow these steps to set up, train, and test the property price prediction model.

### 1\. Initial Setup

#### 1.1. Clone the Repository

```bash
git clone https://github.com/caro6852lq/MachineLearning_PredictPricesProperties.git
cd MachineLearning_PredictPricesProperties
```

#### 1.2. Create and Activate a Virtual Environment

```bash
# On macOS / Linux
python -m venv .venv 
source .venv/bin/activate

# On Windows
python -m venv .venv 
.venv\Scripts\activate
```

#### 1.3. Install Dependencies Using **uv**

This project uses the **uv** package manager (a fast, modern installer) along with the `uv.lock` file for precise dependency management.

**⚠️ Prerequisite:** Make sure you have `uv` installed globally. If not, you can install it using `pip install uv`.

Once inside the virtual environment, sync and install all dependencies:

```bash
uv sync
```

*(This command reads the project dependencies and installs them based on the locked versions in `uv.lock`.)*

-----

### 2\. Model Training

You can train the model using the provided Python script.

#### Train and Save the Model

Run the training script. This process will **train the model** and **save the final model (XGBoost)** as `model_xgb.bin` in the project root.

```bash
python train.py
```

-----

### 3\. Prediction with the Trained Model

Once the `model_xgb.bin` file is available, you can test predictions.

#### Local Prediction Test

Execute the `predict.py` script to load the model and perform a sample prediction.

```bash
python predict.py
```

The script will output the predicted price for the defined sample input data.

-----

### 4\. Deployment

The project is configured to be deployed as a web service, using Docker and Fly.io.

#### 4.1. Build the Docker Image

Ensure that **Docker** is installed on your system.

```bash
docker build -t property-price-predictor .
```

#### 4.2. Run the Container (Local Test)

Run the container and map the port to test the prediction API locally.

```bash
docker run -it --rm -p 9696:9696 property-price-predictor
```

Once the container is running, the API will be available at `http://localhost:9696/predict` for receiving `POST` requests.

#### 4.3. Deploy to Fly.io (Optional)

If you wish to deploy the application online, use the Fly.io CLI, which will utilize the `fly.toml` file for configuration.

```bash
fly deploy
```

-----

🧠 Models and Metrics

Several regression models were tested, including:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

The main evaluation metric used was the Root Mean Squared Error (RMSE), which measures the average difference between predicted and actual prices.



👩‍💻 Author

Carolina Vergara
Midterm Project — Machine Learning Zoomcamp (DataTalks.Club)
📅 October 2025
