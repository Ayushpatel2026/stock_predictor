import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# Load the data - there are 619040 rows and 7 columns
# The columns are: date, open, high, low, close, volume, name
# For each company, there are 5 years of data from 2013 to 2018
data = pd.read_csv('data/all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')

#==============================================================================
# Data Cleaning

#print(data.info())
# there are about 11 rows with null values in the dataset, so we can remove them

# remove NaN values
data = data.dropna()
#print(data.info())

companies = data['name'].unique()
# there are 505 companies in the dataset
num_companies = len(companies)
#print("Number of companies: ", num_companies)

# convert the date column to datetime
data['date'] = pd.to_datetime(data['date'])

#extract the year, month, and day from the date column. This will help us establish temporal relationships
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# encode the company names to integers
# the model may interpret the company names as ordinal data, which is not good
# however we cannot do one-hot encoding because there are 505 companies, which would mean 505 new columns
le = LabelEncoder()
data['name_encoded'] = le.fit_transform(data['name'])

# drop the date column
# data = data.drop(columns=['date'])
#print(data.info())

#==============================================================================
# Data Exploration

# Time series analysis of closing prices
data['date'] = pd.to_datetime(data['date'])
average_close = data.groupby('date')['close'].mean().reset_index()

plt.figure(figsize=(14, 7))
sns.lineplot(data=average_close, x='date', y='close')
plt.title('Average Closing Price Over Time')
plt.show()

# Companywise closing price trend
sample_companies = data['name'].unique()[:5]  # Select 5 sample companies
df_sample = data[data['name'].isin(sample_companies)]

plt.figure(figsize=(14, 7))
sns.lineplot(data=df_sample, x='date', y='close', hue='name')
plt.title('Closing Price Trends for Sample Companies')
plt.show()

numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'year', 'month', 'day', 'name_encoded']
correlation_matrix = data[numeric_columns].corr()
print(data.duplicated())
print(data[data.duplicated()])

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()




