# create_model.py - SIMPLE VERSION
print("🚀 Starting WellnessAI Model Creation...")

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    print("✅ All imports successful!")
    
    # Create simple dataset
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'sleep_quality': np.random.randint(1, 6, n_samples),
        'self_esteem': np.random.randint(10, 31, n_samples),
        'bullying': np.random.randint(1, 6, n_samples),
        'anxiety_level': np.random.randint(0, 11, n_samples),
        'academic_performance': np.random.randint(1, 6, n_samples),
        'depression': np.random.randint(5, 28, n_samples),
        'blood_pressure': np.random.randint(1, 6, n_samples),
        'basic_needs': np.random.randint(1, 6, n_samples),
        'future_career_concerns': np.random.randint(1, 6, n_samples),
        'teacher_student_relationship': np.random.randint(1, 6, n_samples),
        'peer_pressure': np.random.randint(1, 6, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Simple stress calculation
    stress_score = (
        df['anxiety_level'] * 2 +
        (30 - df['self_esteem']) * 0.5 +
        df['depression'] * 1.5 +
        (6 - df['sleep_quality']) * 2
    )
    
    # Create stress levels
    df['stress_level'] = 0  # Default to Low
    df.loc[stress_score > 25, 'stress_level'] = 1  # Moderate
    df.loc[stress_score > 45, 'stress_level'] = 2  # High
    
    # Save dataset
    df.to_csv('StressLevelDataset.csv', index=False)
    print("✅ Dataset created: StressLevelDataset.csv")
    
    # Prepare features
    features = ['sleep_quality', 'self_esteem', 'bullying', 'anxiety_level',
               'academic_performance', 'depression', 'blood_pressure', 'basic_needs',
               'future_career_concerns', 'teacher_student_relationship', 'peer_pressure']
    
    X = df[features]
    y = df['stress_level']
    
    # Simple scaling and model training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    # Save files
    joblib.dump(model, 'rf_classifier_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(features, 'selected_features.pkl')
    
    print("✅ Model files created successfully!")
    print("🎯 You can now run: python app.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please make sure you have Python installed with these packages:")
    print("pip install pandas numpy scikit-learn joblib")
    input("Press Enter to exit...")