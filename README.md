# Diabetes Risk Prediction

## Project Overview
This is a machine learning-based diabetes risk assessment system designed to analyze health indicators and lifestyle data to predict the risk of developing diabetes. The system generates personalized risk assessment reports.

## Data Sources
This project utilizes data from the Behavioral Risk Factor Surveillance System (BRFSS) and NHANES:
- Primary training dataset: diabetes_012_health_indicators_BRFSS2015.csv
- Contains 21 feature variables including high blood pressure, high cholesterol, BMI, smoking status, physical activity, and other health indicators
- Training data quality and diversity enhanced using generative algorithms



## Running Instructions

### Direct Execution
Ensure Python 3.8+ is installed, then follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetes-risk-prediction.git
cd diabetes-risk-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Access in browser: `http://127.0.0.1:5000`

### Docker Execution

1. Ensure Docker is installed, then build the image:
```bash
docker build -t diabetes-risk-prediction .
```

2. Run the container:
```bash
docker run -p 5000:5000 diabetes-risk-prediction
```

3. Access in browser: `http://127.0.0.1:5000`

## Project Structure
- `app.py`: Main Flask application
- `models/`: Trained machine learning models
- `static/`: CSS, JavaScript, and image resources
- `templates/`: HTML template files
- `translations/`: Internationalization files

## Model Evaluation
- Accuracy: 85%+
- AUC Score: 0.82+
- Feature selection efficiency: ~7% performance improvement through genetic algorithm optimization

## Disclaimer
This system is for educational and research purposes only and should not replace professional medical advice. Please consult healthcare professionals for any health concerns.
 

