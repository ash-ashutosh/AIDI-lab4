from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('Fish.csv')
data.columns = ['Species', 'Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']

# Encoding categorical variable 'Species'
label_encoder = LabelEncoder()
data['Species'] = label_encoder.fit_transform(data['Species'])

# Splitting data into features and target variable
X = data.drop('Species', axis=1)
y = data['Species']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])

        new_fish = [[weight, length1, length2, length3, height, width]]
        predicted_species = model.predict(new_fish)
        predicted_species = label_encoder.inverse_transform(predicted_species)
        return render_template('result.html', species=predicted_species[0])

if __name__ == '__main__':
    app.run(debug=True)
