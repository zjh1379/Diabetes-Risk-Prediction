# Models Directory

This directory is used to store PyTorch model files.

## Model Files

Please place the trained PyTorch model `diabetes_model.pt` in this directory.

## Model Structure

The diabetes risk prediction model should accept the following features as input:

1. Age
2. Gender 
3. BMI (Body Mass Index)
4. Systolic Blood Pressure
5. Diastolic Blood Pressure
6. Total Cholesterol Level
7. Fasting Blood Glucose
8. Physical Activity Frequency
9. Daily Fruit and Vegetable Intake
10. Smoking Status
11. Alcohol Consumption Frequency
12. Family History of Diabetes
13. History of Hypertension
14. Related Symptom Indicators
15. Race/Ethnicity Information

The model output should be a single value between 0-1, representing the probability of diabetes risk.

## Model Loading Instructions

If your model uses a custom architecture or requires special loading methods, please modify the model loading code in `app.py`.

If the model is unavailable, the system will use rule-based risk assessment logic as a fallback option.