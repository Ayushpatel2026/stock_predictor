from flask import Flask, render_template, request
import torch
import mlp as mlp
import pandas as pd
import pickle

# Load the model
model = mlp.MLP(input_size=8, hidden_size1=128, hidden_size2=64, hidden_size3=64)
model.load_state_dict(torch.load('models/core_mlp_model.pth'))
model.eval()

# Load the scalers
with open('data/scaler_x.pkl', 'rb') as f:
    scaler_x = pickle.load(f)
with open('data/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Load the label encoder
with open('data/company_name_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def input():
    # Get the input from the form
    date = request.form['date']
    volume = request.form['volume']
    company_name = request.form['company_name']
    high = request.form['high']
    low = request.form['low']
    open_price = request.form['open']

    # turn date into year, month, day
    data = pd.DataFrame({
        'high': [high],
        'open': [open_price],
        'low': [low],
        'volume': [volume],
    })

    dateTime = pd.to_datetime(date)

    data['year'] = dateTime.year
    data['month'] = dateTime.month
    data['day'] = dateTime.day

    # company name needs to be turned into encoded value
    company_encoded = le.transform([company_name])[0]
    data['name_encoded'] = company_encoded

    # scale the input
    scaled_data = scaler_x.transform(data)
    input = torch.tensor(scaled_data, dtype=torch.float32)
    # Make a prediction
    with torch.no_grad():
            output = model(input)
            prediction_scaled = output.numpy()
    # Inverse scale the prediction
    predicted_price = (scaler_y.inverse_transform(prediction_scaled)[0][0]).round(2)
    # Return the prediction
    return render_template('index.html', prediction=predicted_price, input=request.form)

if __name__ == '__main__':
    app.run(port=3000, debug=True)