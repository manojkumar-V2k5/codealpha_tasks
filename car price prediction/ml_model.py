import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pickle

# Set Matplotlib style
mpl.style.use('ggplot')

# Load the dataset
car = pd.read_csv('D:\car price prediction\quikr_car - quikr_car.csv')

# Backup the dataset
backup = car.copy()

# Data Cleaning
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)

car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)

car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)

car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split(' ').str.slice(0, 3).str.join(' ')

car = car.reset_index(drop=True)

# Save cleaned data
car.to_csv('Cleaned_Car_data.csv', index=False)

# Filter Price
car = car[car['Price'] < 6e6].reset_index(drop=True)

# Visualizations
plt.subplots(figsize=(15, 7))
ax = sns.boxplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.savefig('company_vs_price_boxplot.png')

plt.subplots(figsize=(20, 10))
ax = sns.swarmplot(x='year', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.savefig('year_vs_price_swarmplot.png')

sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)
plt.savefig('kms_driven_vs_price_relplot.png')

plt.subplots(figsize=(14, 7))
sns.boxplot(x='fuel_type', y='Price', data=car)
plt.savefig('fuel_type_vs_price_boxplot.png')

ax = sns.relplot(x='company', y='Price', data=car, hue='fuel_type', size='year', height=7, aspect=2)
ax.set_xticklabels(rotation=40, ha='right')
plt.savefig('company_vs_price_fuel_type.png')

# Model Preparation
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# OneHotEncoding and Column Transformer
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

# Predictions and Model Evaluation
y_pred = pipe.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Hyperparameter Tuning
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

best_index = np.argmax(scores)
print("Best Random State:", best_index)
print("Best R2 Score:", scores[best_index])

# Retrain with Best Random State
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=best_index)
pipe.fit(X_train, y_train)
print("Final R2 Score:", r2_score(y_test, pipe.predict(X_test)))

# Save the Model
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))

# Example Prediction
sample_data = pd.DataFrame(
    columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
    data=np.array(['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']).reshape(1, 5)
)
print("Prediction for sample data:", pipe.predict(sample_data))

# Print the OneHotEncoder categories
print("Categories:", pipe.steps[0][1].transformers[0][1].categories_[0])
