import joblib
import numpy as np
import logging
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the Random Forest model
model = joblib.load('best_model_Random Forest.joblib')

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Handle POST request for prediction
        logging.info('Received POST request for prediction.')

        try:
            features = [
                'Vehicle Type Banded',
                'Vehicle Skidding And Overturning',
                'Vehicle Restricted Lane',
                'Location Of Vehicle At First Impact',
                'Vehicle Hit Object In Carriageway',
                'Vehicle Leaving Carriageway',
                'Vehicle Hit Object Off Carriageway',
                'First Point Of Impact',
                'Driver Gender',
                'Casualty Count',
                'Highway Authority',
                'First Road Class',
                'First Road Number',
                'Road Type',
                'Speed Limit',
                'Junction Detail',
                'Junction Control',
                'Second Road Number',
                'Pedestrian Crossing Facilities',
                'Light Conditions',
                'Weather Details',
                'Road Surface Condition',
                'Special Conditions At Site',
                'Carriageway Hazards'
            ]

            input_features = [request.form[feature] for feature in features]
            input_features = np.array(input_features).reshape(1, -1)

            # Example: Predict the severity
            prediction = model.predict(input_features)
            severity = int(prediction[0])

            logging.info(f'Prediction result: {severity}')

            return render_template('result.html', severity=severity)
        
        except Exception as e:
            # Log the error
            logging.error(f'Error during prediction: {str(e)}')

            # Optionally, return an error page or message
            return render_template('error.html', error=str(e))

    else:
        # Handle GET request to display the form
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

