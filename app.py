import os
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g
from werkzeug.utils import secure_filename
# Remove Flask-Babel
# from flask_babel import Babel, gettext as _

# Add direct Babel support
from babel.support import Translations
import gettext

app = Flask(__name__)
app.secret_key = 'diabetes_prediction_key'
# Remove Babel initialization
# babel = Babel(app)

# Supported languages
LANGUAGES = {
    'en': 'English',
    'zh': '中文'
}

# Default language setting
# @babel.localeselector
def get_locale():
    # Use user's language preference if set
    if 'language' in session:
        return session['language']
    # Default to English
    return 'en'

# Simple translation function that replaces Flask-Babel's gettext
translations = {}
def _(text):
    locale = get_locale()
    if locale == 'en':  # No translation needed for English
        return text
        
    if locale not in translations:
        try:
            translations[locale] = gettext.translation(
                'messages', 
                os.path.join(os.path.dirname(__file__), 'translations'),
                languages=[locale]
            )
        except FileNotFoundError:
            return text
    return translations[locale].gettext(text)

# Language switch route
@app.route('/change_language/<language>')
def change_language(language):
    if language in LANGUAGES:
        session['language'] = language
    return redirect(request.referrer or url_for('index'))

# Model path configuration
MODEL_PATH = os.path.join('models', 'diabetes_model.pth')

# Load model if file exists
if os.path.exists(MODEL_PATH):
    try:
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()  # Set to evaluation mode
    except Exception as e:
        print(f"Model loading error: {e}")
        model = None
else:
    model = None
    print(f"Model file not found: {MODEL_PATH}")

# Feature processing variables
feature_means = None
feature_stds = None

# Default standardization parameters for demonstration purposes
# In production, these should be calculated from training data
DEFAULT_FEATURE_MEANS = {
    'age': 45.0, 'height': 165.0, 'weight': 70.0, 
    'systolic': 120.0, 'diastolic': 80.0, 
    'cholesterol': 200.0, 'glucose': 100.0,
    'physical_activity': 2.0, 'fruit_veg': 3.0,
    'smoking': 1.0, 'alcohol': 1.0
}

DEFAULT_FEATURE_STDS = {
    'age': 15.0, 'height': 10.0, 'weight': 15.0, 
    'systolic': 15.0, 'diastolic': 10.0, 
    'cholesterol': 40.0, 'glucose': 25.0,
    'physical_activity': 1.0, 'fruit_veg': 2.0,
    'smoking': 0.8, 'alcohol': 0.8
}

# Risk level thresholds 
RISK_THRESHOLDS = {
    'low': 0.3,     # 0-30% low risk
    'medium': 0.6  # 30-60% medium risk, above 60% high risk
}

# Get localized recommendations based on current language
def get_localized_recommendations():
    recommendations = {}
    
    # English recommendations
    if get_locale() == 'en':
        recommendations = {
            'lifestyle': {
                'low_activity': "Increase weekly physical activity, aiming for at least 150 minutes of moderate-intensity activity per week.",
                'smoking': "Consider quitting smoking, which will significantly reduce various health risks.",
                'alcohol': "Reduce alcohol intake, limit to no more than two drinks per day for men and one drink per day for women."
            },
            'diet': {
                'low_fruit_veg': "Consume at least 5 servings of vegetables and fruits daily.",
                'high_bmi': "Consult a nutritionist for a personalized healthy eating plan."
            },
            'medical': {
                'high_risk': "Please consult a doctor as soon as possible for further diabetes screening and assessment.",
                'medium_risk': "Regular blood glucose monitoring is recommended. Discuss prevention strategies with your doctor.",
                'high_bp': "Monitor your blood pressure regularly and consider consulting a doctor about blood pressure management."
            },
            'resources': [
                {"name": "American Diabetes Association", "url": "https://www.diabetes.org/"},
                {"name": "National Diabetes Prevention Program", "url": "https://www.cdc.gov/diabetes/prevention/index.html"},
                {"name": "Healthy Eating Guidelines", "url": "https://www.myplate.gov/"}
            ]
        }
    # Chinese recommendations
    else:
        recommendations = {
            'lifestyle': {
                'low_activity': "增加每周身体活动，目标为每周至少150分钟中等强度活动。",
                'smoking': "考虑戒烟，这将显著降低多种健康风险。",
                'alcohol': "减少酒精摄入，男性每日不超过两杯，女性每日不超过一杯。"
            },
            'diet': {
                'low_fruit_veg': "每天摄入至少5份蔬菜和水果。",
                'high_bmi': "咨询营养专家获取个性化的健康饮食计划。"
            },
            'medical': {
                'high_risk': "请尽快咨询医生进行进一步的糖尿病筛查和评估。",
                'medium_risk': "建议定期检查血糖水平，并与医生讨论预防策略。",
                'high_bp': "定期监测血压，并考虑咨询医生关于血压管理的建议。"
            },
            'resources': [
                {"name": "美国糖尿病协会", "url": "https://www.diabetes.org/"},
                {"name": "国家糖尿病预防计划", "url": "https://www.cdc.gov/diabetes/prevention/index.html"},
                {"name": "健康饮食指南", "url": "https://www.myplate.gov/"}
            ]
        }
    
    return recommendations

# Home page route
@app.route('/')
def index():
    # Set default language to English
    session['language'] = 'en'
    return render_template('index.html', languages=LANGUAGES)

# Helper function for safe float conversion
def safe_float(value, default=0.0):
    """
    Safely convert a value to float, returning default if conversion fails
    """
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# Prediction results route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data and process inputs
            data = {}
            
            # Basic information
            data['age'] = safe_float(request.form.get('age'), 0)
            data['gender'] = int(request.form.get('gender') == 'male')  # 1 for male, 0 for female
            data['height'] = safe_float(request.form.get('height'), 0)
            data['weight'] = safe_float(request.form.get('weight'), 0)
            
            # Calculate BMI
            if data['height'] > 0 and data['weight'] > 0:
                data['bmi'] = data['weight'] / ((data['height']/100) ** 2)
            else:
                data['bmi'] = 25.0  # Default BMI value
            
            # Health metrics
            data['systolic'] = safe_float(request.form.get('systolic'), DEFAULT_FEATURE_MEANS['systolic'])
            data['diastolic'] = safe_float(request.form.get('diastolic'), DEFAULT_FEATURE_MEANS['diastolic'])
            data['cholesterol'] = safe_float(request.form.get('cholesterol'), DEFAULT_FEATURE_MEANS['cholesterol'])
            data['glucose'] = safe_float(request.form.get('glucose'), DEFAULT_FEATURE_MEANS['glucose'])
            
            # Lifestyle factors
            activity_map = {'none': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
            data['physical_activity'] = activity_map.get(request.form.get('physical_activity', 'moderate'), 2)
            
            data['fruit_veg'] = safe_float(request.form.get('fruit_veg'), DEFAULT_FEATURE_MEANS['fruit_veg'])
            
            smoking_map = {'never': 0, 'former': 1, 'current': 2}
            data['smoking'] = smoking_map.get(request.form.get('smoking', 'never'), 0)
            
            alcohol_map = {'never': 0, 'occasional': 1, 'regular': 2}
            data['alcohol'] = alcohol_map.get(request.form.get('alcohol', 'occasional'), 1)
            
            # Medical history
            data['family_history'] = int(request.form.get('family_history') == 'yes')
            data['hypertension_history'] = int(request.form.get('hypertension_history') == 'yes')
            
            # Process symptoms
            symptoms = request.form.getlist('symptoms')
            data['symptom_thirst'] = 1 if 'increased_thirst' in symptoms else 0
            data['symptom_urination'] = 1 if 'frequent_urination' in symptoms else 0
            data['symptom_weight_loss'] = 1 if 'weight_loss' in symptoms else 0
            data['symptom_fatigue'] = 1 if 'fatigue' in symptoms else 0
            data['symptom_blurred_vision'] = 1 if 'blurred_vision' in symptoms else 0
            
            # Calculate risk using model or fallback method
            if model is not None:
                # Convert to model input format
                features = prepare_features(data)
                input_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
                
                # Use loaded PyTorch model for prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    # Get prediction probability (assuming model outputs probability values)
                    if isinstance(outputs, torch.Tensor) and outputs.numel() == 1:
                        # Single output value case (sigmoid output)
                        prediction = outputs.item()
                    else:
                        # For multi-class cases, use softmax to get class 1 probability
                        prediction = F.softmax(outputs, dim=1)[0, 1].item()
                    
                    risk_percentage = prediction * 100  # Convert to percentage
            else:
                # If model is unavailable, use sample risk calculation
                risk_percentage = calculate_sample_risk(data)
            
            # Determine risk level
            risk_level = get_risk_level(risk_percentage / 100)  # Convert back to 0-1 range
            
            # Generate personalized recommendations
            recommendations = generate_recommendations(data, risk_level)
            
            # Render results page
            return render_template('result.html', 
                                  risk_percentage=risk_percentage,
                                  risk_level=risk_level,
                                  recommendations=recommendations,
                                  user_data=data,
                                  languages=LANGUAGES)
        except Exception as e:
            error_message = "An error occurred during prediction." if get_locale() == 'en' else "预测过程中出错"
            flash(f'{error_message}: {str(e)}', 'danger')
            print(f"Prediction Error: {e}")
            return redirect(url_for('index'))

# Prepare input features for model
def prepare_features(data):
    """
    Convert user input data to feature vector for model prediction.
    Only extracts the first 10 features needed by the model.
    """
    # Features needed by the model (first 10)
    feature_names = [
        'age', 'gender', 'bmi', 'systolic', 'diastolic', 
        'cholesterol', 'glucose', 'physical_activity', 'fruit_veg',
        'smoking'
    ]
    
    features = []
    
    # Build feature vector and standardize
    for feature in feature_names:
        if feature in data:
            # Standardize numeric features
            if feature in DEFAULT_FEATURE_MEANS and feature in DEFAULT_FEATURE_STDS:
                normalized_value = (data[feature] - DEFAULT_FEATURE_MEANS[feature]) / DEFAULT_FEATURE_STDS[feature]
                features.append(normalized_value)
            else:
                features.append(float(data[feature]))
    
    return features

# Rule-based risk calculation (backup for when model is unavailable)
def calculate_sample_risk(data):
    """
    Calculate diabetes risk based on simple rules
    For demonstration only, not for medical use
    """
    base_risk = 10.0  # Starting risk
    
    # Age factor - adjusted for wider age range
    if data['age'] < 18:
        # Children and teenagers have lower baseline risk
        base_risk = max(5.0, base_risk - (18 - data['age']) * 0.3)
    elif data['age'] > 40:
        # Risk increases with age after 40
        base_risk += min(30, (data['age'] - 40) * 0.5)  # Cap the age contribution
    
    # BMI factor
    if data['bmi'] > 25:
        base_risk += (data['bmi'] - 25) * 1.5
    
    # Family history factor
    if data['family_history']:
        base_risk += 15
    
    # Hypertension factor
    if data['hypertension_history'] or data['systolic'] > 140 or data['diastolic'] > 90:
        base_risk += 10
    
    # Blood glucose factor
    if data['glucose'] > 100:
        base_risk += (data['glucose'] - 100) * 0.2
    
    # Symptoms factor
    symptoms_count = (data['symptom_thirst'] + data['symptom_urination'] + 
                      data['symptom_weight_loss'] + data['symptom_fatigue'] + 
                      data['symptom_blurred_vision'])
    base_risk += symptoms_count * 5
    
    # Ensure risk is between 0-100
    return max(0, min(base_risk, 100))

# Determine risk level from risk probability
def get_risk_level(risk_probability):
    """
    Map risk probability to categorical level
    """
    if risk_probability < RISK_THRESHOLDS['low']:
        return 'low'
    elif risk_probability < RISK_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'high'

# Generate personalized health recommendations
def generate_recommendations(data, risk_level):
    """
    Create personalized recommendations based on user data and risk level
    """
    # Get localized recommendation text
    loc_recommendations = get_localized_recommendations()
    
    recommendations = {
        'lifestyle': [],
        'diet': [],
        'medical': [],
        'resources': loc_recommendations['resources']
    }
    
    # Lifestyle recommendations
    if data['physical_activity'] < 2:
        recommendations['lifestyle'].append(loc_recommendations['lifestyle']['low_activity'])
    
    if data['smoking'] == 2:  # Current smoker
        recommendations['lifestyle'].append(loc_recommendations['lifestyle']['smoking'])
    
    if data['alcohol'] == 2:  # Regular drinker
        recommendations['lifestyle'].append(loc_recommendations['lifestyle']['alcohol'])
    
    # Diet recommendations
    if data['fruit_veg'] < 5:
        recommendations['diet'].append(loc_recommendations['diet']['low_fruit_veg'])
    
    if data['bmi'] > 25:
        recommendations['diet'].append(loc_recommendations['diet']['high_bmi'])
    
    # Medical recommendations
    if risk_level == 'high':
        recommendations['medical'].append(loc_recommendations['medical']['high_risk'])
    elif risk_level == 'medium':
        recommendations['medical'].append(loc_recommendations['medical']['medium_risk'])
    
    if data['systolic'] > 130 or data['diastolic'] > 85:
        recommendations['medical'].append(loc_recommendations['medical']['high_bp'])
    
    return recommendations

# Start the application server (use proper WSGI server in production)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 