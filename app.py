from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Debug paths
print("\n" + "="*40)
print("Current working directory:", os.getcwd())
print("Templates folder exists:", os.path.exists('templates'))
print("Index.html exists:", os.path.exists(os.path.join('templates', 'index.html')))
print("="*40 + "\n")

try:
    # Load models
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
    print("All models loaded successfully!")
except Exception as e:
    print(f"\nERROR LOADING MODELS: {str(e)}\n")
    exit()

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
    10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get form data with corrected field names
        features = [
            float(request.form['Nitrogen']),
            float(request.form['Phosphorus']),  # Corrected spelling
            float(request.form['Potassium']),
            float(request.form['Temperature']),
            float(request.form['Humidity']),
            float(request.form['Ph']),
            float(request.form['Rainfall'])
        ]

        # Preprocess features
        single_pred = np.array(features).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Make prediction
        prediction = model.predict(final_features)
        crop_code = prediction[0]

        # Get result
        if crop_code in crop_dict:
            crop = crop_dict[crop_code]
            result = f"{crop} is the best crop to be cultivated"
        else:
            result = "Could not determine the best crop"

        return render_template('index.html', result=result)

    except Exception as e:
        error = f"Error: {str(e)}. Please check your input values."
        return render_template('index.html', result=error)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)