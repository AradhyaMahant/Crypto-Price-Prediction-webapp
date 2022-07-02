# Crypto-Price-Prediction-webapp


## Objective

### Main Objective:
The main objective of our project “Crypto price prediction” is predicting crypto price with the help of historical data and applying deep learning algorithm “LSTM” model which is an extension of Recurrent neural network. As crypto prices are very volatile and are based on many factors which include world news, world trade and etc. For this purpose we are also bringing “Sentiment Analysis” in order to bring polarity to twitter data with the help of TextBlob.

### Sub Objective:
With the growing population and economic crashes, there is a requirement of a prediction system which can predict accurate prices based on historical data of that currency but also due to the volatility of cryptocurrency it can recommend to sell, buy or retain the crypto. The former is done with help of LSTM model and the latter with the help of sentiment analysis of twitter data.


## Methodology

### Reference of software model :
In this project we are using LSTM model of deep learning on time series data to predict the future of crypto for next 30 days. Other than that we have used sentiment analysis on tweets from twitter to analyze weather the price will go up or not. <br /> 


![1](https://user-images.githubusercontent.com/80280041/176985789-85bcecb7-4e43-4cb4-a3cf-8a38e665e735.png)


### Steps explained :


1.Web scraping data from yahoo finance and tweets from twitter.

2.Prepared the data from yahoo to be fed in LSTM model so that it can predict the future values.

3.Splitting the data into train-test split.

4.Training our model on train-set.

5.Testing our model on test-set and checking the performance in form of graphs.

6.Now predicting the future 30 days price of a particular crypto.

7.Then using another approach of twitter sentiment analysis getting tweets of that particular currency.

8.Making dataset of the tweets extracted after filtering and preprocessing them.

9. Then using Textblob we got the polarity and subjectivity of the tweets.

10.Depending on the count of positive, negative or neutral tweets we have recommended the user weather the price will increase, decrease or remain same.


## Results
Tested Currency Ethereum results:

![2](https://user-images.githubusercontent.com/80280041/176986430-8a829163-c8c7-440b-9fab-39a267f67d1b.png)

![3](https://user-images.githubusercontent.com/80280041/176986435-c47014c0-8857-4d59-8ecd-6d17b88f64f1.png)

![4](https://user-images.githubusercontent.com/80280041/176986436-ab7cced7-173f-4c53-a039-b4eecd291f76.png)


![5](https://user-images.githubusercontent.com/80280041/176986440-881b5c32-acff-4a11-8fc0-1f837eeb3e99.png)


![6](https://user-images.githubusercontent.com/80280041/176986445-f9304e7c-885e-446b-a8d9-8ba001ac8f48.png)




![Screenshot 2022-07-02 095535](https://user-images.githubusercontent.com/80280041/176986447-132eda4f-aec7-40af-b839-0b15cebfec95.png)


