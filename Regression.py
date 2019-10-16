import pandas as pd
import quandl
import math, datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
from matplotlib import style
import pickle

style.use('ggplot') #Type of thing I want to use to make my plot look decent
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']] #Gets us the useful columns
#New column for percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
#Daily Move
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#The columns we actually care about
#              price         X        X             X
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close' #This is just a column for our forecasts for close prices

df.fillna(-99999, inplace=True) #Replaces Null data with an outlier

forecast_out = int(math.ceil(0.01*len(df))) #The numbers of day out we want to predict (this is currently 10 days out)
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out) #Shifting the column so each row's label column will be the adj close 10 days into the future

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:] #Stuff we're going to predict against # last 10%
x = x[:-forecast_out] #Up to 90%


df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

#To retrain the model uncomment this out! ############
#clf = LinearRegression(n_jobs=-1) #You can change this type of algorith to anything, n_jobs = number of threads running at a time (1 = linear, -1 = max)
#clf.fit(x_train, y_train) #Trains the model
#Pickle here to save the training data
#with open('linearregression.pickle', 'wb') as f:
    #pickle.dump(clf, f)
######################################################
#Comment this out when retraining
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
####################################################
accuracy = clf.score(x_test, y_test) #Tests the model

forecast_set = clf.predict(x_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan
#This gets us the correct dates 
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #List of our forecast values that are null then it adds the forecast to it

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()
#The reason why you want to pickle is to save our training
