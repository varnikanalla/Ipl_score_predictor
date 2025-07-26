import pandas as pd

import numpy as np

import keras

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, MinMaxScaler



# Load data

df = pd.read_csv('ipl_data.csv')

df = df.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)



# Label encoding

venue_encoder = LabelEncoder()

batting_team_encoder = LabelEncoder()

bowling_team_encoder = LabelEncoder()

striker_encoder = LabelEncoder()

bowler_encoder = LabelEncoder()



df['venue'] = venue_encoder.fit_transform(df['venue'])

df['bat_team'] = batting_team_encoder.fit_transform(df['bat_team'])

df['bowl_team'] = bowling_team_encoder.fit_transform(df['bowl_team'])

df['batsman'] = striker_encoder.fit_transform(df['batsman'])

df['bowler'] = bowler_encoder.fit_transform(df['bowler'])



X = df.drop(['total'], axis=1)

y = df['total']



scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)



# Build and train model

model = keras.Sequential([

    keras.layers.Input(shape=(X.shape[1],)),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dense(216, activation='relu'),

    keras.layers.Dense(1, activation='linear')

])



model.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=1.0))

model.fit(X_scaled, y, epochs=50, batch_size=64)



# Save model and encoders

model.save("ipl_model.h5")

pd.to_pickle(scaler, "scaler.pkl")

pd.to_pickle(venue_encoder, "venue_encoder.pkl")

pd.to_pickle(batting_team_encoder, "bat_encoder.pkl")

pd.to_pickle(bowling_team_encoder, "bowl_encoder.pkl")

pd.to_pickle(striker_encoder, "striker_encoder.pkl")

pd.to_pickle(bowler_encoder, "bowler_encoder.pkl")