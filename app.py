from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data
from model_training import train_models
from dashboard import create_dashboard

app = Flask(__name__)

# Preprocess the data and train models
X_scaled, y, scaler, data, kmeans = preprocess_data('churn.csv')
best_classifier, classifiers = train_models(X_scaled, y)

# Integrate Dash app
dash_app = create_dashboard(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def show_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data with safe defaults
        credit_score = float(request.form.get('credit_score', 650))
        age = int(request.form.get('age', 35))
        tenure = int(request.form.get('tenure', 3))
        balance = float(request.form.get('balance', 0))
        num_of_products = int(request.form.get('num_of_products', 1))
        estimated_salary = float(request.form.get('estimated_salary', 50000))
        has_credit_card = int(request.form.get('has_credit_card', 0))
        is_active_member = int(request.form.get('is_active_member', 0))

        # Prepare input data as DataFrame (match training feature order)
        input_dict = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_credit_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary
        }
        
        input_df = pd.DataFrame([input_dict])  # Convert to DataFrame
        input_scaled = scaler.transform(input_df)  # Scale input

        # Predict churn
        churn_prediction = best_classifier.predict(input_scaled)[0]

        # Predict customer segment
        customer_segment = kmeans.predict(input_scaled)[0]

        # Generate offers
        offers = generate_retention_offers(input_dict, customer_segment)

        return render_template('result.html', churn_prediction=churn_prediction, offers=offers)
    
    except Exception as e:
        return render_template('result.html', error_message=str(e))

def generate_retention_offers(input_dict, customer_segment):
    """Generate retention offers based on user input & cluster segment."""
    offers = []
    
    credit_score = input_dict['CreditScore']
    age = input_dict['Age']
    tenure = input_dict['Tenure']
    balance = input_dict['Balance']
    num_of_products = input_dict['NumOfProducts']
    has_credit_card = input_dict['HasCrCard']
    is_active_member = input_dict['IsActiveMember']
    estimated_salary = input_dict['EstimatedSalary']

    # Offer based on credit score
    if credit_score < 600:
        offers.append('Offering credit score improvement program.')
    elif credit_score > 800:
        offers.append('Providing exclusive VIP banking services.')

    # Offer based on age
    if age > 60:
        offers.append('Special retirement planning services.')

    # Offer based on tenure and balance
    if tenure > 5 and balance > 50000:
        offers.append('Exclusive benefits for long-time customers with high balance.')

    # Offer based on number of products
    if num_of_products > 2:
        offers.append('Upgrade to premium account for free.')

    # Offer based on estimated salary
    if estimated_salary > 100000:
        offers.append('Personalized investment opportunities.')

    # Common offers
    if has_credit_card == 1:
        offers.append('Upgrade credit card with better rewards.')
    if is_active_member == 1:
        offers.append('Loyalty rewards for staying active.')

    # Offers based on customer segment
    if customer_segment == 0:
        offers.append('Special offers for new customers.')
    elif customer_segment == 1:
        offers.append('Incentives for long-term customers.')
    elif customer_segment == 2:
        offers.append('Rewards for high balance customers.')
    elif customer_segment == 3:
        offers.append('Exclusive deals for premium customers.')

    return offers

if __name__ == '__main__':
    app.run(debug=True)
