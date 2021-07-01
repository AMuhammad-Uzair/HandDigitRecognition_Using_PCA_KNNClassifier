"""
Created on Sat Jun 26 01:15:10 2021

@author: muhammaduzair
"""

# Load the libraries used in the code
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from random import randint

#Load the mnist hand digit dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

#Reduce the size of the dataset to perform quick processing
train_data_size=10000
test_data_size=1000
trainX=trainX[0:train_data_size,:]
testX=testX[0:test_data_size,:]
trainY=trainY[0:train_data_size]
testY=testY[0:test_data_size]


Size=trainX.shape
z=Size[0]
row=Size[1]
col=Size[2]
Train=np.array(trainX)
TrainX=Train.reshape(-1,col*row)
TrainX = TrainX/255

#Random number generator to view random image from dataset
Rand_Img_Num=randint(0, z)

#View the image
plt.figure()
plt.imshow(Train[Rand_Img_Num])
plt.title('Example image from training set');
plt.show()


trainMean = TrainX.mean(axis=0)
trainMean=trainMean.transpose()
train_tilde = TrainX-trainMean;

#PCA finding 
pca = PCA(.95)
comp=pca.fit(train_tilde)
TrainingX=pca.transform(TrainX)
Comp=comp.components_.transpose()
F=train_tilde.dot(Comp)

#Reconstruct the image after applying PCA
trainX_Reconstructed=F.dot(Comp.transpose())

Reconstruct=trainX_Reconstructed.reshape((z,col,row))

#View the image after applying PCA
plt.figure(2)
plt.imshow(Reconstruct[Rand_Img_Num])
plt.title("Image after reconstruction")
plt.show()

#Perform same set of operations which were performed on training data
Size1=testX.shape
z1=Size1[0]
row1=Size1[1]
col1=Size1[2]
Test=np.array(testX)
TestX=Test.reshape(-1,col*row)
TestX = TestX/255


testMean = TestX.mean(axis=0)
testMean=testMean.transpose()
test_tilde = TestX-testMean;
test_k = test_tilde.dot(Comp);
New_featured_data_testing = test_k.dot(Comp.transpose())
TestingX=pca.transform(TestX)

#Now train the model using KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)

#Train the classifier (fit the estimator) using the training data
knn.fit(TrainingX, trainY)

#Estimate the accuracy of the classifier on future data, using the test data
print("Accuracy of the Model=",knn.score(TestingX , testY)*100,"%")


#Get a random number for checking test data prediction
Rand_Img_Num=randint(0, z1)

#Select random test image using random number 
Test=New_featured_data_testing[Rand_Img_Num,:]
Testing_Image=Test.reshape(row,col)
#Plot the testing image whose prediction needs to be done
plt.figure()
plt.imshow(Testing_Image)
plt.title("Image Selected For Testing")
plt.show()

#Perform Prediction
Test=TestingX[Rand_Img_Num,:]
Test=Test.reshape((-1, 1))
Test=Test.transpose()
Img=knn.predict(Test)
#Print the results of prediction
print("Prediction:")
print("The Testing Image Contains Digit ",Img)


