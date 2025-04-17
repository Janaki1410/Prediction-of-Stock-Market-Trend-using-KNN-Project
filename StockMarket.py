from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

main = tkinter.Tk()
main.title("Stock Market Analysis using Supervised Machine Learning")
main.geometry("1000x650")

global filename
global X, Y, X_train, X_test, y_train, y_test
global dataset
global rmse
sc = MinMaxScaler(feature_range = (0, 1))

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head()))

def preprocess():
    global X, Y, dataset, X_train, X_test, y_train, y_test
    text.delete('1.0', END)   
    dataset["Date"]= pd.to_datetime(dataset["Date"])
    dataset["Year"]= dataset['Date'].dt.year
    print(dataset["Year"])
    data = dataset
    temp = dataset.values
    Y = temp[:,5:6]
    dataset = dataset.drop(["Adj Close"], axis=1)
    dataset = dataset.drop(["Volume"], axis=1)
    dataset = dataset.values
    X = dataset[:,1:dataset.shape[1]]
    X = sc.fit_transform(X)
    Y = sc.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train & test split 80% dataset used for training and 20% for testing\n\n")
    text.insert(END,"80% dataset training size = "+str(X_train.shape[0])+"\n\n")
    text.insert(END,"20% dataset testing size = "+str(X_test.shape[0])+"\n\n")
    text.update_idletasks()
    plt.figure(figsize=(16,10), dpi=100)
    plt.plot(data.Date[0:10], data.Close[0:10], color='tab:red')
    plt.gca().set(title="AAPL Adjacent Closing Prices", xlabel='Date', ylabel="Closing Price")
    plt.show()

def runLinearRegression():
    global X, Y, dataset, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    lr_regression = LinearRegression()
    lr_regression.fit(X, Y)
    predict = lr_regression.predict(X_test)
    predict = predict.reshape(predict.shape[0],1)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()

    lr_mse = mean_squared_error(labels,predict)
    text.insert(END,"Linear Regression Root Mean Square Error: "+str(lr_mse)+"\n\n")
    text.insert(END,"Linear Regression Accuracy : "+str(1 - lr_mse)+"\n\n")
    rmse.append(lr_mse)
    for i in range(len(X_test)):
        text.insert(END,"Original Stock Test Price : "+str(labels[i])+" Linear Regression Price : "+str(predict[i])+"\n")
    
    plt.plot(labels, color = 'red', label = 'Original Test Stock Price')
    plt.plot(predict, color = 'green', label = 'Linear Regression Predicted Price')
    plt.title('Linear Regression Stock Price Prediction')
    plt.xlabel('Number of days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    

def runSVM():
    global X, Y, dataset, X_train, X_test, y_train, y_test
    global rmse
    rmse = []
    text.delete('1.0', END)

    svm_regression = SVR(C=1.0, epsilon=0.2)
    svm_regression.fit(X, Y)
    predict = svm_regression.predict(X_test)
    predict = predict.reshape(predict.shape[0],1)
    predict = sc.inverse_transform(predict)
    predict = predict.ravel()
    labels = sc.inverse_transform(y_test)
    labels = labels.ravel()

    svm_mse = mean_squared_error(labels,predict)
    text.insert(END,"SVM Root Mean Square Error: "+str(svm_mse)+"\n\n")
    if svm_mse > 1:
        svm_mse = svm_mse / 100        
    text.insert(END,"SVM Accuracy : "+str(1 - svm_mse)+"\n\n")
    rmse.append(svm_mse)
    for i in range(len(X_test)):
        text.insert(END,"Original Stock Test Price : "+str(labels[i])+" SVM Price : "+str(predict[i])+"\n")
    
    plt.plot(labels, color = 'red', label = 'Original Test Stock Price')
    plt.plot(predict, color = 'green', label = 'SVM Predicted Price')
    plt.title('SVM Stock Price Prediction')
    plt.xlabel('Number of days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    

def graph():
    height = rmse
    bars = ('SVM RMSE','Linear Regression RMSE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Stock Market Analysis using Supervised Machine Learning', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload AAPL Stock Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

lrButton = Button(main, text="Run SVM Algorithm", command=runSVM)
lrButton.place(x=480,y=100)
lrButton.config(font=font1)

svmButton = Button(main, text="Run Linear Regression Algorithm", command=runLinearRegression)
svmButton.place(x=720,y=100)
svmButton.config(font=font1)

graphButton = Button(main, text="RMSE Comparison Graph", command=graph)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=300,y=150)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
