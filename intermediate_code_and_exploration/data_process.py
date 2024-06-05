from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from geopy.distance import geodesic
# from datetime import datetime
# from timezonefinder import TimezoneFinder
# from pytz import timezone

# Calculate distance from home function
def calculate_distance(row):
    home_location = (row['lat'], row['long'])
    merch_location = (row['merch_lat'], row['merch_long'])
    return geodesic(home_location, merch_location).miles

# Function to calculate the Haversine distance
# def haversine_distance(lat1, lon1, lat2, lon2):
#     R = 6371  # Earth radius in kilometers
    
#     # Convert degrees to radians
#     lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
#     # Difference in coordinates
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
    
#     # Haversine formula
#     a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
#     c = 2 * np.arcsin(np.sqrt(a))
    
#     # Distance in kilometers
#     distance = R * c
#     return distance

# Function to calculate distance between two points
def calculate_distance2(row1, row2):
    point1 = (row1['lat'], row1['long'])
    point2 = (row2['lat'], row2['long'])
    return geodesic(point1, point2).miles

# # Function to get timezone from latitude and longitude
# def get_timezone_from_lat_long(lat, long):
#     tf = TimezoneFinder()
#     return tf.timezone_at(lat=lat, lng=long)  # Finds the timezone string

# def convert_utc_to_local(utc_dt, lat, long):
#     tz_name = get_timezone_from_lat_long(lat, long)
#     tz = timezone(tz_name)
#     return utc_dt.astimezone(tz)


def process(df):
    # Add new features
    # Rearrange the rows
    df['original_order'] = range(df.shape[0])

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d/%m/%Y %H:%M')
    df['dob'] = pd.to_datetime(df['dob'], format='%d/%m/%Y')

    # df['timestamp'] = pd.to_datetime(df['trans_date_trans_time'], format='%d/%m/%Y %H:%M')
    # df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d/%m/%Y %H:%M')

    df.sort_values(by=['cc_num', 'trans_date_trans_time'], inplace=True)
    # Calculate time difference in hours
    # df['time_since_last_trans'] = df.groupby('cc_num')['trans_date_trans_time'].diff().apply(lambda x: x.total_seconds() / 3600)
    # # Fill NaN values for each user's first transaction
    # df['time_since_last_trans'] = df['time_since_last_trans'].fillna(value=0)
    # Calculate the time difference between transactions
    df['Time_Delta'] = df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds() / 60.0  # Time delta in minutes
    df['Time_Delta'] = df['Time_Delta'].fillna(value=0)
    # Calculate the time difference between current and previous transactions in hours
    # df['time_diff_hours'] = df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds() / 3600
    # df['time_diff_hours'] = df['time_diff_hours'].fillna(0)
    # # Assuming 'Time_Delta' in minutes is already calculated
    # # df['Hourly_Freq'] = df.groupby('cc_num').rolling(window='60min', on='trans_date_trans_time', closed='left')['amt'].count().reset_index(drop=True)
    # # First, ensure the dataframe is sorted as before
    # #df['count_last_24h'] = df.groupby('cc_num')['trans_date_trans_time'].transform(lambda x: x.diff().rolling('12h').count())
    # # Now, to count transactions in the last 24 hours, we initialize a column for counts
    # df['count_last_12h'] = 0

    # for i in df['cc_num'].unique():
    #     # For each credit card number, iterate over transactions
    #     temp = df[df['cc_num'] == i]  # Temporary DataFrame for each credit card
    #     for j in range(len(temp)):
    #         # Calculate the sum of transactions in the last 24 hours
    #         # This is a naive method, and there are more efficient ways to do this with cumulative sums or using pandas' rolling windows with custom criteria
    #         df.loc[temp.index[j], 'count_last_12h'] = temp.iloc[max(0, j-12):j]['time_diff_hours'].count()


    # Group by 'cc_num' and then apply a rolling count with a 24-hour window, shifted to not include the current transaction
    #df['transaction_frequency_24hr'] = df.groupby('cc_num').rolling(window='24h', on='trans_date_trans_time', closed='left').count()

    # Shift the latitude and longitude to get the previous transaction's location
    df['prev_lat'] = df.groupby('cc_num')['merch_lat'].shift(1)
    df['prev_long'] = df.groupby('cc_num')['merch_long'].shift(1)

    # Calculate the distance to the previous transaction
    df['distance_to_prev'] = df.apply(
        lambda row: calculate_distance2(
            {'lat': row['lat'], 'long': row['long']},
            {'lat': row['prev_lat'], 'long': row['prev_long']}
        ) if not pd.isnull(row['prev_lat']) else None,
        axis=1
    )

    # Calculate location consistency as the inverse of the average distance to previous transactions (higher value means more consistency)
    df['location_consistency'] = 100 / df.groupby('cc_num')['distance_to_prev'].transform('mean')

    # Use a rolling window to count distinct merchants for each transaction, excluding the current transaction to avoid lookahead bias
    # df['merchant_variety_24hr'] = df.groupby('cc_num').rolling(window='24h', on='trans_date_trans_time', closed='left')['merchant'].apply(lambda x: x.nunique())

    # Calculate the rolling standard deviation of transaction amounts with a 7-day window for each card
    # df['amount_variability_7days'] = df.groupby('cc_num')['amt'].transform(
    #     lambda x: x.rolling(window='7d', on='trans_date_trans_time').std()
    # )
    
    # Assuming your dataframe is named `data`
    # Convert 'unix_time' to datetime in UTC
    #df['trans_date_trans_time_utc'] = pd.to_datetime(df['unix_time'], unit='s', utc=True)
    # Apply the conversion for each transaction
    # df['local_time'] = df.apply(lambda x: convert_utc_to_local(x['trans_date_trans_time_utc'], x['merch_lat'], x['merch_long']), axis=1)
    # df['local_time'] = pd.to_datetime(df['local_time'])
    # # Extract the hour from 'local_time'
    # df['local_time_hour'] = df['local_time'].dt.hour
    # df['local_day_of_week'] = df['local_time'].dt.dayofweek
    # Time-based features
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

    # df['trans_hour'] = date_time.dt.hour
    # df['trans_day_of_week'] = date_time.dt.dayofweek
    
    # Age of the account holder
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

    # df['trans_dist'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['dist_to_home'] = df.apply(calculate_distance, axis=1)

    # user_avg_amt = df.groupby('cc_num')['amt'].mean().reset_index(name='Avg_Amt')
    # df = df.merge(user_avg_amt, on='cc_num')
    # df['Relative_Amt'] = abs(df['amt'] - df['Avg_Amt']) / df['Avg_Amt']
    # Calculate the historical average transaction amount for each user
    avg_amt_per_user = df.groupby('cc_num')['amt'].transform('mean').rename('avg_amt_per_user')

    # Append this feature to the dataset
    df['amt_relative_avg'] = df['amt'] / avg_amt_per_user
    #df['relative_amt'] = abs(df['amt'] - avg_amt_per_user) / avg_amt_per_user

    df.drop(columns=['trans_date_trans_time', 'lat', 'long', 'merch_lat', 'merch_long'], inplace=True)
    df.drop(columns=['prev_lat', 'prev_long'], inplace=True)

    # Identify categorical columns to encode
    categorical_cols = ['cc_num', 'merchant', 'category', 'gender', 'city', 'state', 'job']

    mappings = {}

    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
        mappings[col] = {label: index for index, label in enumerate(label_encoder.classes_)}

    # Sort the dataset back to its original order
    df.sort_values(by='original_order', inplace=True)
    df.drop(columns='original_order', inplace=True)

    return df, mappings

trainingSet = pd.read_csv("./data/train.csv")
submissionSet = pd.read_csv("./data/test.csv")
train_processed, cat_map = process(trainingSet)
train_processed.drop(columns=['first', 'last', 'street', 'dob', 'zip', 'trans_num', 'unix_time'], inplace=True)

# Merge on Id so that the test set can have feature columns as well
test_df = pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
test_df = test_df.drop(columns=['is_fraud_x'])
test_df = test_df.rename(columns={'is_fraud_y': 'is_fraud'})

# The training set is where the score is not null
train_df = train_processed[train_processed['is_fraud'].notnull()]

# Save the datasets with the new features for easy access later
test_df.to_csv("./data/test_processed4.csv", index=False)
train_df.to_csv("./data/train_processed4.csv", index=False)
