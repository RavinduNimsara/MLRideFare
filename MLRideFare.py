# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopy.distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
!pip install catboost
from catboost import CatBoostClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from google.colab import files
#uploaded = files.upload()

def getDistance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) 
    
    
def getTime(dateTime):
    t=int(dateTime.split(' ')[1].split(':')[0])
    if t<6:
        return 1
    elif t<9:
        return 2
    elif t<17:
        return 3
    else:
        return 4
    
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# path to file you will use for predictions
train_data_path = 'train.csv'
train_data = pd.read_csv(train_data_path)

test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)




train_data['additional_fare'].fillna(train_data['additional_fare'].mean(), inplace=True)

train_data['meter_waiting'].fillna(train_data['meter_waiting'].mean(), inplace=True)
train_data['meter_waiting_fare'].fillna(train_data['meter_waiting_fare'].mean(), inplace=True)
train_data.drop(['meter_waiting_till_pickup'], axis=1)


#train_data=train_data.fillna(train_data.mean(), inplace=True)
train_data=train_data.dropna(axis=0, how='any')
train_data=train_data.reset_index(drop=True)

distance1=[]
pickupTimePeriod1=[]
for i in range(len(train_data['tripid'])):
    try:
        p_lat=train_data['pick_lat'][i]
        p_lon=train_data['pick_lon'][i]
        d_lat=train_data['drop_lat'][i]
        d_lon=train_data['drop_lon'][i]
        cords1=(p_lat,p_lon)
        cords2=(d_lat,d_lon)
        distance1.insert(i,geopy.distance.distance(cords1, cords2).km)
        if train_data['pickup_time'][i] == np.nan:
            pickupTimePeriod1.insert(i,np.nan)
        else:
            pickupTimePeriod1.insert(i,getTime(train_data['pickup_time'][i]))
    except KeyError:
        continue
    
    
train_data['Distance']=distance1#getDistance(train_data['pick_lat'],train_data['pick_lon'],train_data['drop_lat'],train_data['drop_lon'])
train_data['Pickup_period']=pickupTimePeriod1

distance2=[]
pickupTimePeriod2=[]
for i in range(len(test_data['tripid'])):
    try:
        p_lat=test_data['pick_lat'][i]
        p_lon=test_data['pick_lon'][i]
        d_lat=test_data['drop_lat'][i]
        d_lon=test_data['drop_lon'][i]
        cords1=(p_lat,p_lon)
        cords2=(d_lat,d_lon)
        distance2.insert(i,geopy.distance.distance(cords1, cords2).km)
        if test_data['pickup_time'][i] == np.nan:
            pickupTimePeriod2.insert(i,np.nan)
        else:
            pickupTimePeriod2.insert(i,getTime(test_data['pickup_time'][i]))
    except KeyError:
        continue
        
        
#print (len(pickupTimePeriod2))
#print (len(test_data['tripid']))
    
test_data['Distance']=distance2#getDistance(test_data['pick_lat'],test_data['pick_lon'],test_data['drop_lat'],test_data['drop_lon'])
test_data['Pickup_period']=pickupTimePeriod2
train_data['drop_time'] = pd.to_datetime(train_data['drop_time'], format="%m/%d/%Y, %H:%M:%S", errors='coerce')
train_data['pickup_time'] = pd.to_datetime(train_data['pickup_time'], format="%m/%d/%Y, %H:%M:%S", errors='coerce')

train_data['duration'].fillna((train_data['drop_time']-train_data['pickup_time'])/np.timedelta64(1,'s'), inplace=True)



train_data['pickup_hour']=train_data['pickup_time'].dt.hour
train_data['drop_hour']=train_data['drop_time'].dt.hour
train_data['pickup_minute']=train_data['pickup_time'].dt.minute
train_data['drop_minute']=train_data['drop_time'].dt.minute
train_data['pickup_day']=train_data['pickup_time'].dt.day

train_data['driving_time'] = train_data['duration']-train_data['meter_waiting']

test_data['drop_time'] = pd.to_datetime(test_data['drop_time'], format="%m/%d/%Y, %H:%M:%S", errors='coerce')
test_data['pickup_time'] = pd.to_datetime(test_data['pickup_time'], format="%m/%d/%Y, %H:%M:%S", errors='coerce')
test_data['pickup_hour']=test_data['pickup_time'].dt.hour
test_data['drop_hour']=test_data['drop_time'].dt.hour
test_data['pickup_minute']=test_data['pickup_time'].dt.minute
test_data['pickup_day']=test_data['pickup_time'].dt.day
test_data['drop_minute']=test_data['drop_time'].dt.minute
test_data['driving_time'] = test_data['duration']-test_data['meter_waiting']

train_data['f_without_ad_f']=train_data['fare']-train_data['additional_fare']
test_data['f_without_ad_f']=test_data['fare']-test_data['additional_fare']

# The list of columns is stored in a variable called features, 'Distance'
feature1 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup','drop_time','pickup_time',  'pick_lat', 'pick_lon', 'drop_lat', 'drop_lon', 'fare', 'Distance']
feature2 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup', 'Distance', 'fare']
feature3 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'Distance','drop_time','pickup_time', 'fare']
feature4 = ['duration', 'meter_waiting', 'meter_waiting_fare', 'Distance','drop_time','pickup_time','f_without_ad_f']
feature5 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup','drop_time','pickup_time',  'Distance', 'drop_lat', 'drop_lon', 'fare']
feature6 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'Distance', 'fare']
feature7 = ['additional_fare', 'duration', 'meter_waiting', 'meter_waiting_fare', 'Distance','drop_time','pickup_time', 'fare','pickup_hour','drop_hour','driving_time','pickup_day']
features=[feature1,feature2,feature3,feature4,feature5,feature6,feature7]

y = train_data["label"]
X = (pd.get_dummies(train_data[features[6]], dummy_na=True)).fillna(0)
X_test = (pd.get_dummies(test_data[features[6]], dummy_na=True)).fillna(0)


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# To improve accuracy, create a new Random Forest model which you will train on all training data
model = CatBoostClassifier(iterations=200000)#RandomForestClassifier(n_estimators=100,max_depth=40, random_state=1)
model.fit(train_X,train_y)
predictions=model.predict(val_X)
acc=f1_score(val_y, predictions)
print (acc)
#fit on all data from the training data
#model.fit(X, y)

# make predictions which we will submit. 
predictions = model.predict(X_test)

# #create output
output = pd.DataFrame({'tripid': test_data.tripid, 'prediction': predictions})
output.to_csv('sample_submission.csv', index=False)

print ('Done')
