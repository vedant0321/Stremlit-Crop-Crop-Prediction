# Crop Prediction Using Machine Learning

Welcome to our Crop Prediction app, powered by advanced machine learning algorithms! Our platform leverages cutting-edge technology to help farmers and agricultural stakeholders make informed decisions about their crops, ensuring better yield and efficient resource management.

## Features

- **Accurate Yield Predictions:** Our machine learning models analyze a vast array of data, including historical weather patterns, soil conditions, and crop performance, to provide accurate yield predictions.
- **Personalized Insights:** Get tailored recommendations for your specific farm conditions, helping you decide the best crops to plant and the optimal times for planting and harvesting.
- **Real-Time Data Analysis:** Stay updated with real-time data analysis and forecasts, allowing you to react promptly to changing conditions.
- **Resource Optimization:** Efficiently manage resources such as water, fertilizers, and pesticides based on precise predictions, reducing waste and increasing sustainability.

## Methodology

Our crop prediction model leverages an ensemble approach based on critical agricultural attributes collected in the year 2023. The dataset comprises 2202 entities, each characterized by attributes related to soil and environmental conditions, specifically NPK levels (Nitrogen, Phosphorus, Potassium), pH, humidity, and rainfall. The target variable is the type of crop among 21 possible crops.

### Model Selection

An **ensemble** model is chosen to leverage the strengths of multiple learning algorithms. The ensemble model combines the predictions of the following base models:
- **Random Forest Classifier:** An ensemble of decision trees, which reduces overfitting and improves accuracy.
- **Gradient Boosting Classifier:** Sequentially builds models to correct the errors of the previous models, enhancing performance.
- **Voting Classifier:** The base models are combined using a voting classifier, which aggregates their predictions to improve overall accuracy.

## How to Run the App

### Prerequisites

- Python 3.x
- Streamlit
- Pandas
- Numpy
- Matplotlib
- Plotly
- streamlit-navigation-bar
- scikit-learn
- pickle

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/crop-prediction-app.git
   cd crop-prediction-app

2. pip install -r requirements.txt

### Contact

For inquiries, feedback, or collaboration opportunities, please feel free to reach out to us:

Name: Vedant Birewar
Email: vedantbirewar@gmail.com
LinkedIn: Vedant Birewar
GitHub: vedant0321
'''
[link to deployment](https://crop-ai.streamlit.app/)'''