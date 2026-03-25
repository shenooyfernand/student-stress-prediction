from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='.')

print("🚀 Loading WellnessAI ML Model...")

try:
    model = joblib.load('rf_classifier_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_data = pd.DataFrame([{
            'sleep_quality': int(data['sleep_quality']),
            'self_esteem': int(data['self_esteem']),
            'bullying': int(data['bullying']),
            'anxiety_level': int(data['anxiety_level']),
            'academic_performance': int(data['academic_performance']),
            'depression': int(data['depression']),
            'blood_pressure': int(data['blood_pressure']),
            'basic_needs': int(data['basic_needs']),
            'future_career_concerns': int(data['future_career_concerns']),
            'teacher_student_relationship': int(data['teacher_student_relationship']),
            'peer_pressure': int(data['peer_pressure'])
        }])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        stress_levels = {0: 'Low', 1: 'Moderate', 2: 'High'}
        prediction_label = stress_levels.get(prediction, 'Unknown')

        return jsonify({'prediction': prediction_label, 'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

# ADD THESE NEW ROUTES FOR IMAGES
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('.', filename)

@app.route('/<filename>.jpg')
def serve_jpg(filename):
    return send_from_directory('.', f'{filename}.jpg')

@app.route('/<filename>.png')
def serve_png(filename):
    return send_from_directory('.', f'{filename}.png')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    print("\n🌐 WellnessAI Server Starting at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    app.run(debug=True, port=5000)