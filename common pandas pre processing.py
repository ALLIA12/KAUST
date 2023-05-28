import pandas as pd
import numpy as np

# reading the csv file itself
df = pd.read_csv('iris.csv')
feature_names = df.keys()
X = df.iloc[:, 0:4].to_numpy().astype(np.float32)
y = df.iloc[:, 4].to_numpy()
y[y == 'Iris-setosa'] = 0
y[y == 'Iris-versicolor'] = 1
y[y == 'Iris-virginica'] = 2
y = y.astype(np.int64)
N = y.shape[0]

# drop values
df.dropna(inplace=True)

# fill values
df.fillna(0, inplace=True)

# fill with mean or median
df.fillna(df.mean(), inplace=True)  # fill with mean
df.fillna(df.median(), inplace=True)  # fill with median

# interpolation filll
df.interpolate(inplace=True)

# one hot encoding
encoded_df = pd.get_dummies(df['column_name'])

# label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['column_name'] = le.fit_transform(df['column_name'])

# scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['column_name'] = scaler.fit_transform(df[['column_name']])

# min max scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['column_name'] = scaler.fit_transform(df[['column_name']])

# outliers
from scipy import stats

z_scores = stats.zscore(df['column_name'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
new_df = df[filtered_entries]

# date hanelding
df['year'] = df['date_column'].dt.year
df['month'] = df['date_column'].dt.month
df['day'] = df['date_column'].dt.day
df['time'] = df['date_column'].dt.time

# duplicate rows
df.drop_duplicates(inplace=True)

# duplicate value in a column
df['column_name'].drop_duplicates(inplace=True)

# boolean to integer
df['bool_column'] = df['bool_column'].astype(int)

# checking data type of all column
df.dtypes
