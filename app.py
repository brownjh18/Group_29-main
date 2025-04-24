from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from stable_baselines3 import PPO
import gym
from gym import spaces
import os

app = Flask(__name__)  # Initialize the Flask web application

# Custom Gym Environment for Bovine Respiratory Disease prediction
class BovineRespiratoryDiseaseEnv(gym.Env):
    def __init__(self, data):
        super(BovineRespiratoryDiseaseEnv, self).__init__()
        self.data = data
        self.current_step = 0
        num_features = data.shape[1] - 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(37,), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step, :-1].values.astype(np.float32)
    
    def step(self, action):
        observation = self.data.iloc[self.current_step, :-1].values.astype(np.float32)
        reward = 10 if action == 1 and self.data.iloc[self.current_step]['BRD_Total'] > 0 else -5
        self.current_step += 1
        done = self.current_step >= len(self.data)
        return observation, reward, done, {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        data = pd.read_excel(file)
        required_columns = ['BRD_Total']
        if not all(col in data.columns for col in required_columns):
            return jsonify({'error': 'Missing required columns in the data'}), 400
        
        data['BRD_Total'] = data['BRD_Total'].fillna(0).astype(int)
        data = data.fillna(0)
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                except Exception:
                    pass

        for col in data.select_dtypes(include=['datetime64']).columns:
            data[col] = data[col].apply(lambda x: x.timestamp() if pd.notna(x) else 0)

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].apply(lambda x: x / (x.max() if x.max() > 0 else 1))

        env = BovineRespiratoryDiseaseEnv(data)
        model = PPO('MlpPolicy', env, learning_rate=0.0001, verbose=1)
        model.learn(total_timesteps=10000)
        model.save("brd_model.zip")
        return jsonify({'message': 'Model trained and saved successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        data = request.json
        if not data or "features" not in data:
            return jsonify({'error': 'Please provide feature values'}), 400
        
        if not os.path.exists("brd_model.zip"):
            return jsonify({'error': 'Model not found. Train the model first!'}), 400

        model = PPO.load("brd_model.zip")
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)
        action, _ = model.predict(features, deterministic=True)

        return jsonify({'prediction': 'Intervene' if action == 1 else 'No intervention needed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Use dynamic port for deployment (e.g., on Render)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Use PORT from environment if available
    app.run(host='0.0.0.0', port=port, debug=True)
