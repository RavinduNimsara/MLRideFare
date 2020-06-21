# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train_data_path = '../input/dataset/train.csv'
train_data = pd.read_csv(train_data_path)

test_data_path = '../input/dataset/test.csv'
test_data = pd.read_csv(test_data_path)


train_data['drop_time'] = pd.to_datetime(train_data['drop_time'], errors='coerce')
train_data['pickup_time'] = pd.to_datetime(train_data['pickup_time'], errors='coerce')
test_data['drop_time'] = pd.to_datetime(test_data['drop_time'], errors='coerce')
test_data['pickup_time'] = pd.to_datetime(test_data['pickup_time'], errors='coerce')

train_data['trip_time'] = train_data.drop_time - train_data.pickup_time
test_data['trip_time'] = test_data.drop_time - test_data.pickup_time


train_data['trip_time'] = train_data['trip_time'].dt.total_seconds()
test_data['trip_time'] = test_data['trip_time'].dt.total_seconds()

# Create target object and call it y
y = train_data["label"]

# The list of columns is stored in a variable called features
features = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup', 'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare', 'trip_time']

# Create X and X_test buy using dummies
X = (pd.get_dummies(train_data[features], dummy_na=True)).fillna(0)
X_test = (pd.get_dummies(test_data[features], dummy_na=True)).fillna(0)


# To improve accuracy, create a new Random Forest model which you will train on all training data
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1)

#fit on all data from the training data
model.fit(X, y)

# make predictions which we will submit. 
predictions = model.predict(X_test)

# #create output
output = pd.DataFrame({'tripid': test_data.tripid, 'prediction': predictions})
output.to_csv('sample_submission.csv', index=False)


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
