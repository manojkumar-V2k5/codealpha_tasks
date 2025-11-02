import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model/LinearRegressionModel.pkl', 'rb'))

# Function to encode plots as base64 strings
def encode_plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/')
def home():
    # Load the car data
    car = pd.read_csv('data/Cleaned_Car_data.csv')
    
    # Get unique values for dropdowns and convert to lists
    companies = car['company'].unique().tolist()  # Convert to list
    fuel_types = car['fuel_type'].unique().tolist()  # Convert to list
    years = car['year'].unique().tolist()  # Convert to list
    years.sort(reverse=True)  # Sort years in descending order

    return render_template('index.html', companies=companies, fuel_types=fuel_types, years=years)

@app.route('/get_car_names', methods=['POST'])
def get_car_names():
    selected_company = request.form['company']
    car = pd.read_csv('data/Cleaned_Car_data.csv')
    car_names = car[car['company'] == selected_company]['name'].unique().tolist()
    return jsonify(car_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    name = request.form['name']
    company = request.form['company']
    year = int(request.form['year'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = request.form['fuel_type']

    sample_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                               columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Predict the price
    prediction = model.predict(sample_data)
    
    # Reload the car data to get dropdown values again
    car = pd.read_csv('data/Cleaned_Car_data.csv')
    car_names = car['name'].unique().tolist()
    companies = car['company'].unique().tolist()
    years = car['year'].unique().tolist()
    years.sort(reverse=True)  # Sort years in descending order
    fuel_types = car['fuel_type'].unique().tolist()

    return render_template('index.html', prediction_text=f'Predicted Price: {prediction[0]:.2f} INR',
                           companies=companies, fuel_types=fuel_types, years=years, car_names=car_names)

@app.route('/graphs')
def graphs():
    # Generate all plots
    car = pd.read_csv('data/Cleaned_Car_data.csv')

    # Boxplot for Company vs Price
    plt.figure(figsize=(15, 7))
    ax = sns.boxplot(x='company', y='Price', data=car)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    boxplot_image = encode_plot_to_base64(plt)

    # Scatter plot for Year vs Price
    plt.figure(figsize=(20, 10))
    ax = sns.swarmplot(x='year', y='Price', data=car)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    swarmplot_image = encode_plot_to_base64(plt)

    # Kms Driven vs Price plot
    sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)
    relplot_image = encode_plot_to_base64(plt)

    # Boxplot for Fuel Type vs Price
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='fuel_type', y='Price', data=car)
    boxplot_fuel_image = encode_plot_to_base64(plt)

    return render_template('graphs.html', boxplot_image=boxplot_image,
                           swarmplot_image=swarmplot_image,
                           relplot_image=relplot_image,
                           boxplot_fuel_image=boxplot_fuel_image)

if __name__ == "__main__":
    app.run(debug=True)