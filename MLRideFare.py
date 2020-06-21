# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

##----------------
## Remove trip time because duration is present
##----------------

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopy.distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

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

#train_data['trip_time'] = train_data.drop_time - train_data.pickup_time
#test_data['trip_time'] = test_data.drop_time - test_data.pickup_time


#train_data['trip_time'] = train_data['trip_time'].dt.total_seconds()
#test_data['trip_time'] = test_data['trip_time'].dt.total_seconds()

# Create target object and call it y

distance1=[]
for i in range(len(train_data['tripid'])):
    p_lat=train_data['pick_lat'][i]
    p_lon=train_data['pick_lon'][i]
    d_lat=train_data['drop_lat'][i]
    d_lon=train_data['drop_lon'][i]
    cords1=(p_lat,p_lon)
    cords2=(d_lat,d_lon)
    distance1.insert(i,geopy.distance.distance(cords1, cords2).km)
    
train_data['Distance']=distance1

distance2=[]
for i in range(len(test_data['tripid'])):
    p_lat=test_data['pick_lat'][i]
    p_lon=test_data['pick_lon'][i]
    d_lat=test_data['drop_lat'][i]
    d_lon=test_data['drop_lon'][i]
    cords1=(p_lat,p_lon)
    cords2=(d_lat,d_lon)
    distance2.insert(i,geopy.distance.distance(cords1, cords2).km)
    
test_data['Distance']=distance2
#########modify no need of for loop
#fare_without_additionalfare1=[]
#for i in range(len(train_data['tripid'])):
#    fare_without_additionalfare1.insert(i,train_data['fare'][i]-train_data['additional_fare'][i])
train_data['f_without_ad_f']=train_data['fare']-train_data['additional_fare']
test_data['f_without_ad_f']=test_data['fare']-test_data['additional_fare']

#fare_without_additionalfare2=[]
#for i in range(len(test_data['tripid'])):
#    fare_without_additionalfare2.insert(i,test_data['fare'][i]-test_data['additional_fare'][i])
#test_data['f_without_ad_f']=fare_without_additionalfare2
    

# The list of columns is stored in a variable called features

feature1 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup',  'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare']
feature2 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup', 'Distance', 'fare']
feature3 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'Distance', 'fare']
feature4 = ['duration', 'meter_waiting', 'meter_waiting_fare', 'Distance','f_without_ad_f']

features=[feature1,feature2,feature3,feature4]

X = (pd.get_dummies(train_data[feature1], dummy_na=True)).fillna(0)
y = train_data["label"]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1)
model.fit(train_X, train_y)
predictions = model.predict(val_X)

mae = mean_absolute_error(predictions, val_y)

print ("Test train done\n")
mx_lf_nodes=[5, 8, 10, 50, 100, 500]
performance=[]
c=0
min_mae=float("inf")
min_mae_c=0
f=0
n=0
for i in features: 
    for j in mx_lf_nodes:
        print (c)
        X = (pd.get_dummies(train_data[i], dummy_na=True)).fillna(0)
        y = train_data["label"]
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
        model = RandomForestClassifier(n_estimators=100, max_depth=j, random_state=1)
        model.fit(train_X, train_y)
        predictions = model.predict(val_X)
        mae = mean_absolute_error(predictions, val_y)
        performance.insert(c,[f,n,mae])
        if min_mae>mae:
            min_mae=mae
            min_mae_c=c
        c+=1
        n+=1
    n=0
    f+=1
        
print ("Minimum mae is %f feature %d & max lf nodes %d"%(min_mae,performance[min_mae_c][0],performance[min_mae_c][1]))

X = (pd.get_dummies(train_data[features[performance[min_mae_c][0]]], dummy_na=True)).fillna(0)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestClassifier(n_estimators=100, max_depth=performance[min_mae_c][1], random_state=1)
model.fit(train_X, train_y)

X_test = (pd.get_dummies(test_data[features[performance[min_mae_c][0]]], dummy_na=True)).fillna(0)
predictions = model.predict(X_test)

# #create output
output = pd.DataFrame({'tripid': test_data.tripid, 'prediction': predictions})

output.to_csv('sample_submission.csv', index=False)
print ('Done')


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
