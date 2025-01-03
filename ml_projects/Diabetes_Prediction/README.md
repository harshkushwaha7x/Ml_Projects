# Diabetes Prediction Project

This project is designed to predict whether a person is likely to have diabetes based on a set of medical parameters. It uses machine learning models and provides a user-friendly web interface for predictions.

---

## Project Structure

### 1. **Directories**
- **`.idea/`**: Contains project configuration files (specific to the development environment).
- **`static/`**: Contains static assets like CSS, JavaScript, and images used in the web interface.
- **`templates/`**: Contains HTML templates for rendering the user interface of the web application.

### 2. **Files**
- **`app.py`**: The main Flask application file that handles routing and integrates the trained model for predictions.
- **`diabetes_prediction_dataset.csv`**: The dataset used to train and test the machine learning model. It includes features like glucose levels, BMI, age, etc.
- **`model.ipynb`**: A Jupyter Notebook used for data exploration, preprocessing, and training the machine learning model.
- **`model.py`**: Contains the code for training the model and saving it for later use.
- **`requirements.txt`**: Lists the Python dependencies required to run the project.

---

## Features
- **Machine Learning Model**: Predicts the likelihood of diabetes based on user-provided data.
- **Web Interface**: A Flask-based application that allows users to input their medical parameters and view predictions.
- **Dataset Utilization**: Uses a structured dataset to train the machine learning model.

---

## How to Set Up the Project

### 1. Clone the Repository
```bash
https://github.com/yourusername/your-repo.git
```

### 2. Install Dependencies
Install all required Python libraries using:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask application by running:
```bash
python app.py
```

### 4. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

---

## Dataset
The dataset, `diabetes_prediction_dataset.csv`, contains the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (0: No Diabetes, 1: Diabetes)

---

## Technologies Used
- **Programming Language**: Python
- **Framework**: Flask
- **Machine Learning**: Scikit-learn, Pandas
- **Frontend**: HTML, CSS, JavaScript (in `static/` and `templates/` folders)

---

## Future Enhancements
- Integrate additional features for better prediction accuracy.
- Deploy the application to a cloud platform for wider accessibility.
- Add user authentication for personalized tracking.

---

## Contributors
Feel free to contribute to this project by creating pull requests or submitting issues.

---

## License
This project is open-source and available under the [MIT License](LICENSE).
