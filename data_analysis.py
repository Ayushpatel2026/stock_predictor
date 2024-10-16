import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#==============================================================================
# Data Cleaning

def load_and_clean_data(filepath):
    """
    Load and clean the stock data from the given filepath.
    Returns the cleaned DataFrame.
    """
    # Load the data - there are 619040 rows and 7 columns
    # The columns are: date, open, high, low, close, volume, name
    # For each company, there are 5 years of data from 2013 to 2018
    data = pd.read_csv(filepath, delimiter=',', on_bad_lines='skip')

    # Remove NaN values
    data = data.dropna()

    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Extract year, month, and day from the date column
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    # encode the company names to integers
    # the model may interpret the company names as ordinal data, which is not good
    # however we cannot do one-hot encoding because there are 505 companies, which would mean 505 new columns
    le = LabelEncoder()
    data['name_encoded'] = le.fit_transform(data['name'])

    return data

#==============================================================================
# Data Exploration

def plot_closing_price_over_time(data):
    """
    Plot the average closing price over time.
    """
    average_close = data.groupby('date')['close'].mean().reset_index()
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=average_close, x='date', y='close')
    plt.title('Average Closing Price Over Time')
    plt.show()

def plot_companywise_trend(data):
    """
    Plot the closing price trends for a sample of companies.
    """
    sample_companies = data['name'].unique()[:5]  # Select 5 sample companies
    df_sample = data[data['name'].isin(sample_companies)]
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_sample, x='date', y='close', hue='name')
    plt.title('Closing Price Trends for Sample Companies')
    plt.show()

def plot_correlation_heatmap(data, numeric_columns):
    """
    Plot the correlation heatmap of the numeric features.
    """
    correlation_matrix = data[numeric_columns].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()

#==============================================================================
# Data Preprocessing if needed

#==============================================================================
# Split data into training, validation, and test sets

def split_data(data, training):
    """
    Split the data into training, validation, and test sets.
    Returns the split sets.
    """
    X = data[training]  # Features
    y = data['close']  # Target variable (closing price)

    # First split: train + validation, test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Second split: train, validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test



