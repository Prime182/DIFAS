from flask import Flask, request, jsonify, render_template
import requests
import os
import glob
import time
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objects as go
from twilio.rest import Client
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import numpy as np
import time
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
# Twilio credentials (replace with your own)
account_sid = os.environ.get('ACCOUNT_SID')
auth_token = os.environ.get('AUTH_TOKEN')
twilio_phone_number = os.environ.get('TWILIO_PHONE_NUMBER')

# print(f'account_sid: {account_sid} \n auth_token: {auth_token} \n twilio_phone_number: {twilio_phone_number}')
country_code = '+91'  # Replace it with client phone number including code +91
message = ""
status = ""

# Initialize Twilio client
client = Client(account_sid, auth_token)
# file_path = 'static/forecast.png'

# Function to get latitude and longitude from location name
def get_lat_lon(location_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {'name': location_name, 'count': 1}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            latitude = data['results'][0]['latitude']
            longitude = data['results'][0]['longitude']
            return latitude, longitude
    return None, None

# Function to create message
def create_message(discomfort_hours):
    global message
    messages = []
    
    if discomfort_hours['No discomfort']:
        messages.append(f"No discomfort expected at the following hours: {', '.join(discomfort_hours['No discomfort'])}.")
    if discomfort_hours['Mild discomfort']:
        messages.append(f"Mild discomfort expected at the following hours: {', '.join(discomfort_hours['Mild discomfort'])}.")
    if discomfort_hours['Moderate discomfort']:
        messages.append(f"Moderate discomfort expected at the following hours: {', '.join(discomfort_hours['Moderate discomfort'])}.")
    if discomfort_hours['Severe discomfort']:
        messages.append(f"Severe discomfort expected at the following hours: {', '.join(discomfort_hours['Severe discomfort'])}.")
    if discomfort_hours['Very Severe discomfort']:
        messages.append(f"Very Severe discomfort expected at the following hours: {', '.join(discomfort_hours['Very Severe discomfort'])}.")

    if not messages:
        messages.append("No discomfort levels predicted in the next 24 hours.")
    message = '\n'.join(messages)
    # print("final message",message)

# Function to send SMS alert
def send_sms_alert(phone):
    global status
    client_phone_number = country_code + phone
    # print(client_phone_number)
    # Combine all messages into one and send SMS
    
    # print(message)
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=client_phone_number
    )

    status+=''.join('SMS alert sent!')
    print("SMS alert sent!")
def create_di_plot(df, timestamp, forecast, location): 
    # Calculate date range: last two weeks until the end of tomorrow
    two_weeks_ago = datetime.now() - timedelta(days=14)
    next_day = datetime.now() + timedelta(days=1)

    # Filter actual DI data to the last two weeks
    df_last_two_weeks = df[df['Time'] >= two_weeks_ago]
    
    # Filter forecasted DI data to the same range (from two weeks ago to the end of tomorrow)
    forecast_filtered = forecast[(forecast['ds'] >= two_weeks_ago) & (forecast['ds'] <= next_day)]

    file_name = f'di_{location}_{timestamp}.png'
    file_path = os.path.join('static', file_name)

    # Plot the original DI data and the forecast with a black theme
    plt.figure(figsize=(14, 7))
    plt.style.use('dark_background')

    # Plot actual DI values from the last two weeks
    plt.plot(df_last_two_weeks['Time'], df_last_two_weeks['Discomfort Index'], label='Actual DI', color='white')

    # Plot forecasted DI values from the last two weeks up to the next day
    plt.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label='Forecasted DI', color='orange')

    # Set labels, title, and legend
    plt.xlabel('Time', fontsize=12, color='white')
    plt.ylabel('Discomfort Index (°C)', fontsize=12, color='white')
    plt.title('Discomfort Index (DI) Trends and Forecast', fontsize=14, color='white')

    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.grid(True, color='gray')

    plt.legend()
    plt.savefig(file_path)
    plt.close()

    return file_name


def create_time_series_plot(df, timestamp, location):
    template = "plotly_dark"
    # Filter data to the last two weeks
    df_last_two_weeks = df[df['Time'] >= (datetime.now() - timedelta(days=14))]

    # Continue with the same logic for plotting using df_last_two_weeks
    trace_temp = go.Scatter(
        x=df_last_two_weeks['Time'],
        y=df_last_two_weeks["Temperature (°C)"],
        mode='lines',
        name='Temperature (°C)',
        line=dict(color='blue')
    )

    trace_humidity = go.Scatter(
        x=df_last_two_weeks['Time'],
        y=df_last_two_weeks["Humidity (%)"],
        mode='lines',
        name='Humidity (%)',
        line=dict(color='green')
    )

    trace_discomfort = go.Scatter(
        x=df_last_two_weeks['Time'],
        y=df_last_two_weeks["Discomfort Index"],
        mode='lines',
        name='Discomfort Index',
        line=dict(color='slateblue')
    )

    # Create the figure and add traces
    fig = go.Figure()

    fig.add_trace(trace_temp)
    fig.add_trace(trace_humidity)
    fig.add_trace(trace_discomfort)

    # Make all traces initially invisible except the first one
    for trace in fig.data:
        trace.visible = False
    fig.data[0].visible = True  # Temperature plot is visible initially

    # Define buttons for toggling between the plots
    buttons = [
        dict(
            label='Temperature (°C)',
            method='update',
            args=[{'visible': [True, False, False]},
                {'title': 'Distribution of Temperatures'}],
            # Button styling
        ),
        dict(
            label='Humidity (%)',
            method='update',
            args=[{'visible': [False, True, False]},
                {'title': 'Distribution of Humidity'}],
            # Button styling
        ),
        dict(
            label='Discomfort Index',
            method='update',
            args=[{'visible': [False, False, True]},
                {'title': 'Distribution of Discomfort Index'}],
            # Button styling
        )
    ]

    updatemenus = [
        dict(
            type="buttons",
            direction="down",
            buttons=buttons,
            showactive=True,
            # x=0.17,  # x-position of the buttons
            # y=1.15,  # y-position of the buttons
            # xanchor='left',
            # yanchor='top',
            font=dict(color='black'),
        )
    ]

    # Update layout to include buttons and set the initial title
    fig.update_layout(
        updatemenus=updatemenus,
        template=template,
        title='Distribution of Temperatures',  # Initial title
        autosize=True
    )

    # fig.update_layout(updatemenus=dict(buttons = dict(font = dict( color = "black"))))
    

    # Update axis labels and grid settings
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # plt_div = plot(fig, output_type='div', include_plotlyjs=False)
    file_name = f'ts_{location}_{timestamp}.html'
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)

    # Read the HTML content from the file
    # with open(plot_html_file, 'r') as f:
    #     plot_html_content = f.read()
    # plot(fig, filename='static/plot.html', auto_open=False)
    return file_name

# Function to create a heatmap trace for a given feature
def create_heatmap_trace(data, feature):
    heatmap_data = data.pivot_table(values=feature, index='Date', columns='Hour', aggfunc='mean')
    heatmap_text = heatmap_data.round(2).astype(str).values

    heatmap = go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='thermal',
        text=heatmap_text,
        hoverinfo='text'
    )

    annotations = []
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            annotations.append(
                go.layout.Annotation(
                    x=heatmap_data.columns[j],
                    y=heatmap_data.index[i],
                    text=heatmap_text[i][j],
                    showarrow=False,
                    font=dict(color='white' if heatmap_data.values[i, j] < (heatmap_data.values.max() / 2) else 'black')
                )
            )

    return heatmap, annotations

def create_heatmap(df, timestamp, location):
    # Filter data for the last two weeks
    two_weeks_ago = datetime.now() - timedelta(days=14)
    df_filtered = df[df['Time'] >= two_weeks_ago]

    # Prepare data for heatmaps
    df_filtered['Date'] = df_filtered['Time'].dt.date
    df_filtered['Hour'] = df_filtered['Time'].dt.hour

    # Create heatmap traces and annotations for each feature
    features = ['Temperature (°C)', 'Humidity (%)', 'Discomfort Index']
    titles = ['Heatmap of Temperature by Date and Hour', 'Heatmap of Humidity by Date and Hour', 'Heatmap of Discomfort Index by Date and Hour']

    heatmap_traces = []
    annotations_list = []

    for feature in features:
        heatmap, annotations = create_heatmap_trace(df_filtered, feature)
        heatmap_traces.append(heatmap)
        annotations_list.append(annotations)

    # Initialize figure with all traces but only show the first one
    fig = go.Figure(data=heatmap_traces)

    # Set initial visibility
    for i, trace in enumerate(fig.data):
        trace.visible = (i == 0)

    # Set the initial x-axis range to show only 12 hours
    initial_x_range = [-0.5, 11.5]  # Shows only the first 12 points

    fig.update_layout(
        title=titles[0],
        xaxis=dict(nticks=24, title='Hour', range=initial_x_range),
        yaxis=dict(title='Date'),
        annotations=annotations_list[0],
        template='plotly_dark',
        autosize=True
    )

    # Create dropdown buttons for switching between heatmaps
    dropdown_buttons = [
        dict(
            args=[{'visible': [j == i for j in range(len(features))]},
                  {'annotations': annotations_list[i],
                   'title': titles[i]}],
            label=title,
            method='update'
        )
        for i, title in enumerate(titles)
    ]

    # Add dropdown menu to layout
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction='down',
                showactive=True,
                x=1.15,  # Positioning the button to the right
                y=1.15,  # Positioning the button at the top
                font=dict(color='black')
            )
        ],
        xaxis=dict(title='Hour', range=initial_x_range),
        yaxis=dict(title='Date'),
    )

    file_name = f'heatmap_{location}_{timestamp}.html'
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)

    return file_name


# Function to create histograms for the last two weeks
def create_histogram(df, timestamp, location):
    # Filter data for the last two weeks
    two_weeks_ago = datetime.now() - timedelta(days=14)
    df_filtered = df[df['Time'] >= two_weeks_ago]

    # Create histograms with bar outlines (edges) for different features
    trace_temp = go.Histogram(
        x=df_filtered["Temperature (°C)"],
        nbinsx=80,
        marker=dict(color='blue', line=dict(width=1, color='black')),
        opacity=0.75,
        name="Temperature (°C)"
    )

    trace_humidity = go.Histogram(
        x=df_filtered["Humidity (%)"],
        nbinsx=50,
        marker=dict(color='green', line=dict(width=1, color='black')),
        opacity=0.75,
        name="Humidity (%)"
    )

    trace_discomfort_index = go.Histogram(
        x=df_filtered["Discomfort Index"],
        nbinsx=80,
        marker=dict(color='slateblue', line=dict(width=1, color='black')),
        opacity=0.75,
        name="Discomfort Index"
    )

    # Create a figure and add all traces, but make them initially invisible
    fig = go.Figure()

    fig.add_trace(trace_temp)
    fig.add_trace(trace_humidity)
    fig.add_trace(trace_discomfort_index)

    # Make all traces initially invisible except for the first one
    for trace in fig.data:
        trace.visible = False
    fig.data[0].visible = True  # Only the Temperature plot is visible initially

    # Define buttons for toggling between the plots
    buttons = [
        dict(label='Temperature (°C)',
             method='update',
             args=[{'visible': [True, False, False]},
                   {'title': 'Histogram of Temperature (°C)'}]),
        dict(label='Humidity (%)',
             method='update',
             args=[{'visible': [False, True, False]},
                   {'title': 'Histogram of Humidity (%)'}]),
        dict(label='Discomfort Index',
             method='update',
             args=[{'visible': [False, False, True]},
                   {'title': 'Histogram of Discomfort Index'}]),
    ]

    # Update the layout to include buttons and set the initial title
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="down", buttons=buttons, showactive=True, font=dict(color='black'))],
        template="plotly_dark",
        title="Histogram of Temperature (°C)",  # Initial title
        autosize=True
    )

    # Update axis labels and grid settings
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    file_name = f'histogram_{location}_{timestamp}.html'
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)

    return file_name

def create_discomfort_levels_forecast_plot(forecast, timestamp, location):
    file_name = f'discomfort_levels_forecast_{location}_{timestamp}.png'
    file_path = os.path.join('static', file_name)
    
    # Filter the forecast for the next day from now
    start_time = datetime.now()
    end_time = start_time + timedelta(days=1)
    forecast_next_day = forecast[(forecast['ds'] >= start_time) & (forecast['ds'] < end_time)]

    # Apply the classification function to the forecast
    forecast_next_day['Discomfort Level'] = forecast_next_day['yhat'].apply(classify_discomfort)

    # Count the occurrences of each discomfort level
    discomfort_counts = forecast_next_day['Discomfort Level'].value_counts()

    # Plot the bar plot
    fig0 = plt.figure()
    discomfort_counts.plot(kind='bar', color=['green', 'yellow', 'orange', 'red'])
    plt.style.use('dark_background')
    plt.title('Forecasted Discomfort Levels for the Next Day')
    plt.xlabel('Discomfort Level')
    plt.xticks(rotation=0, color='white')
    plt.ylabel('Frequency')
    plt.show()
    plt.savefig(file_path)
    plt.close()

    return file_name

def cleanup_old_images_html_files():
    # List all PNG files in the static directory
    image_files = glob.glob('static/*.png')
    html_files = glob.glob('static/*.html')
    
    # Sort the files by creation time (oldest first)
    image_files.sort(key=os.path.getctime)
    html_files.sort(key=os.path.getctime)

    # Retain the plots of the last 2 location
    while len(image_files) > 4:
        oldest_file = image_files.pop(0)
        os.remove(oldest_file)
        print(f"Deleted old image: {oldest_file}")
    while len(html_files) > 6:
        oldest_file = html_files.pop(0)
        os.remove(oldest_file)
        print(f"Deleted old html: {oldest_file}")

def classify_discomfort(di):
    if di < 21:
        return "No discomfort"
    elif 21 <= di < 24:
        return "Mild discomfort"
    elif 24 <= di < 27:
        return "Moderate discomfort"
    elif 27 <= di < 29:
        return "Severe discomfort"
    else:
        return "Very Severe discomfort"

@app.route('/get_forecast', methods=['POST'])
def get_forecast():
    global message, status
    
    # Get JSON input data
    data = request.get_json()
    location = data.get('location')
    phone = data.get('phone')

    # Get latitude and longitude
    latitude, longitude = get_lat_lon(location)

    if not latitude or not longitude:
        return jsonify({'error': 'Location not found'}), 404

    # Calculate date range for the last two months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)

    # Set up the API parameters
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'hourly': 'temperature_2m,relative_humidity_2m',
        'timezone': 'Asia/Kolkata'
    }

    # Make the API request
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to retrieve data'}), response.status_code

    # Extract data from API response
    data = response.json()
    time_data = data['hourly']['time']
    temperature_data = data['hourly']['temperature_2m']
    humidity_data = data['hourly']['relative_humidity_2m']

    # Create DataFrame
    df = pd.DataFrame({
        'Time': time_data,
        'Temperature (°C)': temperature_data,
        'Humidity (%)': humidity_data
    })
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Calculate the Discomfort Index
    df['Discomfort Index'] = df['Temperature (°C)'] - 0.55 * (1 - 0.01 * df['Humidity (%)']) * (df['Temperature (°C)'] - 14.5)
    df['Discomfort Level'] = df['Discomfort Index'].apply(classify_discomfort)

    # Prepare the DataFrame for LSTM model
    lstm_df = df[['Time', 'Discomfort Index']].rename(columns={'Time': 'ds', 'Discomfort Index': 'y'})
    lstm_df.set_index('ds', inplace=True)

    # Step 1: Data Preprocessing for LSTM
    values = lstm_df['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    # Step 2: Creating sliding windows of data
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])  # previous 'window_size' values
            y.append(data[i, 0])  # the next value (target)
        return np.array(X), np.array(y)

    window_size = 60
    X, y = create_sequences(scaled_values, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshaping to (samples, time steps, features)

    # Step 3: Model Creation
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=False), input_shape=(window_size, 1)))
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 4: Model Training
    epochs = 10
    model.fit(X, y, epochs=epochs, batch_size=32)

    # Step 5: Calculate R-squared for training data predictions
    from sklearn.metrics import r2_score

    # Predict on training data
    predicted_train_values = model.predict(X)

    # Inverse transform the predicted values to the original scale
    predicted_train_values = scaler.inverse_transform(predicted_train_values)
    actual_train_values = scaler.inverse_transform(y.reshape(-1, 1))

    # Calculate R-squared score
    r_squared = r2_score(actual_train_values, predicted_train_values)

    # Log the R-squared value
    print(f"R-squared value on training data: {r_squared}")

    # Step 6: Forecasting next 24 hours
    last_sequence = scaled_values[-window_size:]  # Get the last 'window_size' data points for prediction
    forecast_horizon = 24
    future_forecast = []

    for _ in range(forecast_horizon):
        last_sequence = last_sequence.reshape((1, window_size, 1))
        next_value = model.predict(last_sequence)[0, 0]
        future_forecast.append(next_value)
        
        # Update the sequence by appending the predicted value and removing the first value
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_value]]], axis=1)

    # Step 7: Inverse transform the predicted values to the original scale
    future_forecast = np.array(future_forecast).reshape(-1, 1)
    future_forecast = scaler.inverse_transform(future_forecast)

    # Prepare the forecasted DataFrame
    future_dates = pd.date_range(start=lstm_df.index[-1], periods=forecast_horizon + 1, freq='H')[1:]
    forecast_df = pd.DataFrame(future_forecast, index=future_dates, columns=['yhat'])
    
    # Convert index to 'ds' column for compatibility
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={'index': 'ds'}, inplace=True)

    # Continue with the existing logic using `forecast_df` instead of `forecast`
    timestamp = int(time.time())
    di_plot_file_name = create_di_plot(df, timestamp, forecast_df, location)
    ts_plot_file_name = create_time_series_plot(df, timestamp, location)
    heatmap_file_name = create_heatmap(df, timestamp, location)
    histogram_file_name = create_histogram(df, timestamp, location)
    discomfort_levels_forecast_file_name = create_discomfort_levels_forecast_plot(forecast_df, timestamp, location)
    
    # Cleanup old images and HTML files
    cleanup_old_images_html_files()

    # Initialize dictionary to store discomfort levels by hour
    discomfort_hours = {
        "No discomfort": [],
        "Mild discomfort": [],
        "Moderate discomfort": [],
        "Severe discomfort": [],
        "Very Severe discomfort":[]
    }

    # Categorize forecasted hours by discomfort level
    for index, row in forecast_df.iterrows():
        discomfort_level = classify_discomfort(row['yhat'])
        if row['ds'] > datetime.now() and row['ds'] <= datetime.now() + timedelta(days=1):
            discomfort_hours[discomfort_level].append(row['ds'].strftime('%Y-%m-%d %H:%M'))

    create_message(discomfort_hours)
    
    # Send SMS alert with categorized discomfort levels
    send_sms_alert(phone)

    final_message = message
    sms_status = status
    print(f"final_message: {final_message}")

    # Reset message and status for future requests
    message = ""
    status = ""

    return jsonify({
        'location': location,
        'latitude': latitude,
        'longitude': longitude,
        'forecast': forecast_df[['ds', 'yhat']].to_dict(orient='records'),
        'r_squared': r_squared,
        'discomfort_hours': discomfort_hours,
        'sms_message': final_message,
        'sms_status': sms_status,
        'di_plot_file': di_plot_file_name,
        'ts_plot_file': ts_plot_file_name,
        'heatmap_file': heatmap_file_name,
        'histogram_file': histogram_file_name,
        'discomfort_levels_forecast_file': discomfort_levels_forecast_file_name
    })

# @app.route('/forecast_plot')
# def forecast_plot():
#     return app.send_static_file('forecast.png')

@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)