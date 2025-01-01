from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("diabetes_model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if it's a form submission
        if request.form:
            # Collect features from form
            features = [
                float(request.form['gender']),
                float(request.form['age']),
                float(request.form['hypertension']),
                float(request.form['heart_disease']),
                float(request.form['smoking_history']),
                float(request.form['bmi']),
                float(request.form['HbA1c_level']),
                float(request.form['blood_glucose_level'])
            ]

            # Convert to numpy array
            features = np.array([features])
            
            # Get prediction
            prediction = model.predict(features)
            
            # Determine result
            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
            
            # Render template with prediction
            return render_template("index.html", prediction_text=f"Prediction Result: {result}")
        
        # Check if it's a JSON submission
        elif request.is_json:
            # Get the data from the JSON body
            data = request.get_json()

            # Extract features from the JSON body
            features = [
                data.get('gender'),
                data.get('age'),
                data.get('hypertension'),
                data.get('heart_disease'),
                data.get('smoking_history'),
                data.get('bmi'),
                data.get('HbA1c_level'),
                data.get('blood_glucose_level')
            ]

            # Convert to numpy array
            features = np.array([features])
            
            # Get prediction
            prediction = model.predict(features)
            
            # Return JSON response
            return jsonify({
                "prediction": int(prediction[0]),
                "result": "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
            })
        
        else:
            # If neither form nor JSON, return an error
            return render_template("index.html", 
                                   prediction_text="Error: Invalid submission method")
    
    except Exception as e:
        # Handle any errors during prediction
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)