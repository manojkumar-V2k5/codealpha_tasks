from flask import Flask, render_template, request
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset and train model once at startup
iris = sns.load_dataset('iris')
X = iris.drop('species', axis=1)
y = iris['species']

# Encode species as numerical labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train classifier
model = RandomForestClassifier()
model.fit(X, y_encoded)

def create_flower_theme_plot():
    # Set seaborn style with a flower theme
    sns.set(style='darkgrid', palette='pastel')

def generate_plot_image(plot_func):
    # Create a plot and return its base64-encoded PNG image
    buf = io.BytesIO()
    plt.figure()
    plot_func()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize')
def visualize():
    # Load Iris dataset
    df = sns.load_dataset('iris')

    # Apply flower theme
    create_flower_theme_plot()

    # Plot 1: Petal Length Distribution
    def plot_petal_length():
        sns.histplot(data=df, x='petal_length', hue='species', multiple='stack')
        plt.title('Petal Length Distribution by Species')

    petal_length_img = generate_plot_image(plot_petal_length)

    # Plot 2: Sepal Width Distribution
    def plot_sepal_width():
        sns.histplot(data=df, x='sepal_width', hue='species', multiple='stack')
        plt.title('Sepal Width Distribution by Species')

    sepal_width_img = generate_plot_image(plot_sepal_width)

    # Plot 3: Pairplot
    def plot_pairplot():
        # Pairplot is not directly saved as image, so save to buffer
        plt.figure()
        sns.pairplot(df, hue='species')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf

    buf = plot_pairplot()
    pairplot_b64 = base64.b64encode(buf.read()).decode('utf-8')

    images = {
        'petal_length': petal_length_img,
        'sepal_width': sepal_width_img,
        'pairplot': pairplot_b64
    }

    return render_template('visualize.html', images=images)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('prediction.html')
    elif request.method == 'POST':
        # Get input features from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Debug: print input features
        print(f"Input features: {sepal_length}, {sepal_width}, {petal_length}, {petal_width}")

        # Prepare input for prediction
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Debug: check the shape and content
        print(f"Input array: {input_features}")

        # Predict
        pred_encoded = model.predict(input_features)[0]
        pred_species = le.inverse_transform([pred_encoded])[0]

        # Debug: predicted class
        print(f"Predicted class: {pred_species}")

        # Show result
        return render_template('result.html', species=pred_species)

if __name__ == '__main__':
    app.run(debug=True)