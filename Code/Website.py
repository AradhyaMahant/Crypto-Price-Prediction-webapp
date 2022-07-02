## Part1 Time Series Prediction(LSTM)

# In[]:


#importing the required libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

#%%website data
st.title('\t CRYPTO PRICE PREDICTER')
st.subheader("Time series data analysis Prediction:  ")

user_input = st.text_input('Enter a currency for prediction','BTC')


#%%
# 1.Loading the financial data(web scraping)

crypto = user_input
against = 'INR'


start =dt.datetime(2020,1,1)
end = dt.datetime.now()

df = web.DataReader(f'{crypto}-{against}','yahoo',start,end)
df.head(10)

st.subheader("Data From 2020 - Till today")
st.write(df.head(10))

# In[]:
    
# 2.Preparing the data(Data Pre-processing)

#removing unnecessary coloumns
df = df.reset_index()
df = df.drop(['Adj Close'], axis = 1)


# In[]:

df.head()

# In[]:
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize = (10,4))
plt.plot(df.Close)
st.pyplot(fig)

# In[]:

df1 = df.reset_index()['Close']

# In[]:


# 3.Downscaling the data (MinMax scaler)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


print(df1)

# In[]:
    
# 4.Splitting the dataset into train and test split

training_size = int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

print(training_size,test_size)
# In[]:
#converting an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[ i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[]:

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[]:

print(X_train.shape)
print(y_train.shape)


# In[]:

print(X_test.shape)
print(y_test.shape)


# In[]:

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)



# In[]:
# 5.LSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM 

model = Sequential()
model.add(LSTM(units = 50,return_sequences = True,input_shape = (100,1)))

model.add(LSTM(units = 50,return_sequences = True))

model.add(LSTM(units = 50))

model.add(Dense(units = 1))

model.compile(loss='mean_squared_error',optimizer='adam')


# In[]:

model.summary()

# In[]:

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=64,verbose=1)

# In[]:
#7. Testing our model on test dataset
#Prediction and Performance Matrix
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# In[]:
#Upscaling the data

#back to orirginal form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# In[]:

# RMSE performance metrics for training dataset
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# RMSE for test data
math.sqrt(mean_squared_error(y_test,test_predict))


# In[]:

# Plotting 
fig2=plt.figure(figsize=(12,6)) 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(scaler.inverse_transform(df1),'b',label ='Original Price')
plt.plot(trainPredictPlot,'orange',label ='Train Predicted Price')
plt.plot(testPredictPlot,'r',label='Test Predicted Price')
plt.legend()
st.subheader("Predicted price vs Original(Past)")
st.pyplot(fig2)

#comparing the orginal close prices to model predicted prices
data2 = pd.concat([df .iloc[-203:].copy(),pd.DataFrame(test_predict,columns=['Close_predicted'],index=df .iloc[-203:].index)], axis=1)
data2 = data2.reset_index()
data2 = data2.drop(['High','Low','Open','Volume','index'], axis = 1)
st.write(data2.tail(10))

# In[]:

# 7.Predicting future 30 days price
tlen=len(test_data)

x_input=test_data[tlen-100:].reshape(1,-1)
print(x_input.shape)
# In[]:

#storing data collected into list
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[]:

#logic for prediction

lst_output=[]
n_steps = 100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else :
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
# In[]:

day_new=np.arange(1,101)
day_pred = np.arange(101,131)

# In[]:

datalen=len(df1)

# In[]:
fig3=plt.figure(figsize=(12,6)) 
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(day_new,scaler.inverse_transform(df1[datalen-100:]),label='Past prices')
plt.plot(day_pred,scaler.inverse_transform(lst_output),label='Future Price')
plt.legend()
st.subheader("Predicted price for next 30 days")
st.pyplot(fig3)

#%%
## Part 2 :Sentiment Analysis Prediction

#importing required libs
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re


# 1.Extracting tweets about the news according to the choice of cryptocurrency entered by the user.
st.subheader("Prediction Based on the news from twitter: ")
# In[]:

consumer_key = "bpc3PoE0WYDPLANrjF6bxKU6V"
consumer_secret = "inY3jXiXFKuzdexomqa76h2xY9scEZhpDIchsT5pbMqZXx8g7l"
access_token =  "1481145969391841280-QpYxOkxap8cJAb50DhrAldzRLVgdL6"
access_token_secret = "eGBJD360SXzbG77j4U78JEAUOFHUuXvMCuYkmTLedU1iE"


# In[]:

#authenticate to use the keys
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# In[]:


#Extracting specific tweets of crypto news from twitter
search_words = "#" + user_input + " -filter:retweets"


# In[]:

tweets = tweepy.Cursor(api.search_tweets,q=search_words,lang="en").items(200)
tweets

tweets_copy=[]
for i in tweets:
    tweets_copy.append(i)
print("Total Tweets fetched: ",len(tweets_copy))


# In[]:
print(tweets_copy[0])

#2. Makin a dataframe out of the tweets extracted

# In[]:

tweets_df = pd.DataFrame()
for tweets in tweets_copy:
    hashtags=[]
    try:
        for hashtag in tweets.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweets.id, tweet_mode='extended').full_text
    except:
        pass
    tweets_df = tweets_df.append(pd.DataFrame({'date': tweets.created_at,
                                              'news': api.get_status(id = tweets.id,tweet_mode = 'extended').full_text,
                                              'hashtags': [hashtags if hashtags else None]}))
    tweets_df = tweets_df.reset_index(drop=True)
    
# 3. Cleaning the text from the tweets 

# In[]:


#Clean the news

#creating a function to clean the news
def CleanTxt(text):
    text = re.sub(r'@[A-za-z0-9]+','', text) #removed @mentions
    text = re.sub(r'#', '', text)#Removing the '#' symbol
    text = re.sub(r'RT[\s]+', '', text) #Removing RT
    text = re.sub(r'https?:\/\/\S+', '', text)# Remove the hyper link
    
    return text

tweets_df['news'] = tweets_df['news'].apply(CleanTxt)


# In[]:


#converting news to lower case
tweets_df['news'] = tweets_df['news'].str.lower()


# 4. Checking for sentiments(positive or negative or neutral)

# In[]:


#creating a function to get the subjectivity 
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

tweets_df['Subjectivity']= tweets_df['news'].apply(getSubjectivity)
tweets_df['Polarity'] = tweets_df['news'].apply(getPolarity)


# In[]:

#plot the word cloud 
allWords = ' '.join( [twts for twts in tweets_df['news']])
wordCloud = WordCloud(width = 500, height= 300, random_state= 21, max_font_size=99).generate(allWords)

plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis('off')
plt.show()


# In[]:


#computing negative, neutral and positive analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
tweets_df['Analysis'] = tweets_df['Polarity'].apply(getAnalysis)

st.subheader("DataSet for analyzing sentiments:")
st.write(tweets_df.head(10))

# In[]:


#plot to show polarity and subjectivity
fig4=plt.figure(figsize=(12,6))
for i in range(0, tweets_df.shape[0]):
    plt.scatter(tweets_df['Polarity'][i],tweets_df['Subjectivity'][i], color='Blue')
    
st.subheader('Sentiment Analysis')    
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
st.pyplot(fig4)


# 5.Getting % of positive , negative and neutral tweets.

# In[]:


ptweets = tweets_df[tweets_df.Analysis == 'Positive']
ptweets = ptweets['news']

PosPer = round( (ptweets.shape[0]/tweets_df.shape[0]) *100, 1)


# In[]:


ntweets = tweets_df[tweets_df.Analysis == 'Negative']
ntweets = ntweets['news']

NegPer=round( (ntweets.shape[0]/tweets_df.shape[0]) *100, 1)


# In[]:

neutweets = tweets_df[tweets_df.Analysis == 'Neutral']
neutweets = neutweets['news']

NeuPer = round( (neutweets.shape[0]/tweets_df.shape[0]) *100, 1)


# In[]:


#showing these values in form of a bar chart

tweets_df['Analysis'].value_counts()

fig5 =plt.figure(figsize=(10,4))
#plot and visuwalize the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
tweets_df['Analysis'].value_counts().plot(kind='bar')
st.pyplot(fig5)

# 6.Comapring %'s of sentiments and predicting accordingly

# In[]:

#predicting the prices result based upon the sentiments
result = ' '
def Prediction(positive, negative, neutral):
  
    if (positive >= negative) and (positive >= neutral):
        result = "Currency prices are likey to increase"
  
    elif (negative >= positive) and (negative >= neutral):
        result = "Currency prices are likey to decrease"
    else:
        result = "Currency prices might remain same"
          
    return result

st.subheader("According to these graphs :")
st.subheader(Prediction(PosPer, NegPer, NeuPer))


#%%                                                                        THE END





