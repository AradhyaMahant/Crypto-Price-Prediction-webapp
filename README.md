# Crypto-Price-Prediction-webapp

Technical concept
The reference algorithm that we used for this project is a deep learning model LSTM also know as Long short-term memory. LSTM is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can process not only single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDSs (intrusion detection systems).
A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. 

Motivation
During 2017 the worth of a solitary Bitcoin expanded 2000% going from $863 on January 9, 2017 to a high of $17,550 on December 11, 2017. By about two months after the fact, on February 5, 2018, the cost of a solitary Bitcoin had been more than split with a worth of $79,643. The promising innovation behind cryptographic forms of money, the blockchain, makes all things considered, digital currencies will keep on being utilized in some limit, and that their utilization will develop.
Therefore, looking at the boom in cryptocurrency we decided to start our work on this project.


Problem Statement
The problem for predicting prices of crypto is that first you have to train a RNN which has vanishing and exploding gradient problem so in order to solve that we have used LSTM on top of that taking a long sequence of data to feed this model is not so easy. The prices may vary due to many factors and taking each and every factor into account is a tough job so we have used two factors mainly past prices of currency and the tweets about the related currency.
 




Area of application
This Project can definitely be helpful for:
 •                  Those people who want to invest in cryptocurrencies.
•                  People who just want to know the current crypto scene.
•                  Regular customers who make and receive payments in cryptocurrency (dark web).
 People can easily see the predicted price and invest. A wide selection of currencies helps new investors to start from a cheaper cryptocurrency rather than directly aiming for cryptocurrencies like Bitcoin and Ethereum. This project will be of significant help for daily based crypto merchants. 

Data Inputs
To solve the problem of predicting cryptocurrency price changes several different data sources are considered as possible inputs to the model. The first input considered is sentiment analysis of collected tweets about Bitcoin or Ethereum.
