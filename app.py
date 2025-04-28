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
        """
        Initialize the environment with the provided data.
        """
        super(BovineRespiratoryDiseaseEnv, self).__init__()
        self.data = data  # Store the dataset
        self.current_step = 0  # Keep track of the current step
        num_features = data.shape[1] - 1  # Number of features (excluding the target)
        self.action_space = spaces.Discrete(2)  # Action space: 2 discrete actions (Intervene or No intervention)
        self.observation_space = spaces.Box(low=0, high=1, shape=(37,), dtype=np.float32)  # Observation space: 37 features scaled to [0, 1]
    
    def reset(self):
        """
        Reset the environment to the initial state (start of the dataset).
        """
        self.current_step = 0
        return self.data.iloc[self.current_step, :-1].values.astype(np.float32)  # Return the initial feature values
    
    def step(self, action):
        """
        Perform one step in the environment, based on the chosen action.
        """
        observation = self.data.iloc[self.current_step, :-1].values.astype(np.float32)  # Current features
        # Reward: +10 if action=1 (intervene) and BRD_Total > 0 (BRD detected), -5 for other cases
        reward = 10 if action == 1 and self.data.iloc[self.current_step]['BRD_Total'] > 0 else -5
        self.current_step += 1  # Move to the next step
        done = self.current_step >= len(self.data)  # Check if the dataset is exhausted
        return observation, reward, done, {}  # Return the new observation, reward, done flag, and additional info

# Route for the home page (render the index.html template)
@app.route('/')
def index():
    return render_template('index.html')

# Route to train the model using uploaded data
@app.route('/train', methods=['POST'])
def train():
    try:
        file = request.files['file']  # Get the uploaded file from the request
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400  # Handle missing file error
        
        data = pd.read_excel(file)  # Read the Excel file into a DataFrame
        required_columns = ['BRD_Total']  # Define the required columns for the dataset (BRD_Total is the target)
        if not all(col in data.columns for col in required_columns):
            return jsonify({'error': 'Missing required columns in the data'}), 400  # Validate the presence of required columns
        
        data['BRD_Total'] = data['BRD_Total'].fillna(0).astype(int)  # Fill missing values in BRD_Total with 0 and convert to int
        # Identify and convert date-like columns
        data = data.fillna(0)  # Fill missing values in the dataset with 0
        for col in data.columns:
            if data[col].dtype == 'object':  # Check for columns with string (object) type
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')  # Convert to datetime (invalid dates will be set to NaT)
                except Exception:
                    pass  # Skip columns that fail to convert to datetime

        # Handle datetime columns (convert to Unix timestamp)
        for col in data.select_dtypes(include=['datetime64']).columns:
            data[col] = data[col].apply(lambda x: x.timestamp() if pd.notna(x) else 0)  # Convert valid dates to Unix timestamps and invalid dates to 0

        # Normalize the numeric columns to scale them between 0 and 1
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].apply(lambda x: x / (x.max() if x.max() > 0 else 1))  # Prevent division by 0
        
        # Initialize the custom Gym environment with the processed data
        env = BovineRespiratoryDiseaseEnv(data)
        model = PPO('MlpPolicy', env, learning_rate=0.0001, verbose=1)  # Initialize the PPO model with the MLP policy
        model.learn(total_timesteps=10000)  # Train the model for 10,000 timesteps
        model.save("brd_model.zip")  # Save the trained model to a file
        return jsonify({'message': 'Model trained and saved successfully!'})  # Return a success message
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message if something goes wrong

# Route to predict using a manually provided set of features
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        data = request.json  # Get the JSON data from the request
        if not data or "features" not in data:
            return jsonify({'error': 'Please provide feature values'}), 400  # Handle missing features error
        
        if not os.path.exists("brd_model.zip"):  # Check if the model file exists
            return jsonify({'error': 'Model not found. Train the model first!'}), 400  # Handle missing model error

        model = PPO.load("brd_model.zip")  # Load the trained model
        features = np.array(data["features"], dtype=np.float32).reshape(1, -1)  # Convert input features to a numpy array and reshape
        action, _ = model.predict(features, deterministic=True)  # Get the model's prediction (action)

        return jsonify({'prediction': 'Intervene' if action == 1 else 'No intervention needed'})  # Return the prediction result
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message if something goes wrong

# Start the Flask web server on port 8000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
