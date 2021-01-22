# DALT7011 ML-Assignment 1
# Student ID: 19137568
# Student Name: Luca Di Carlo

######################################################
# Load function provided by brendan o'connor - gist.github.com/39760 - anyall.org
######################################################
# INSERT THE PATH TO THE WORKING DIRECTORY OF THE 4 DATA FILES
setwd("INSERT YOUR SYSTEM PATH HERE/data/")

# load function
load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train_ <<- load_image_file('train-images-idx3-ubyte')
  test_ <<- load_image_file('t10k-images-idx3-ubyte')
  
  train_$y <<- load_label_file('train-labels-idx1-ubyte')
  test_$y <<- load_label_file('t10k-labels-idx1-ubyte')  
}

show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

######################################################
# Dimensionality Reduction (PCA)
######################################################
library(ggplot2)
# load data
load_mnist()

# Exploratory Analysis
summary(train_$x)
train_max <- max(train_$x)
train_min <- min(train_$x)
test_max <- max(test_$x)

# check for missing values
sum(is.na(train_$x))

# standardize the data (do not center or scale, yields better results)
digits_X <- train_$x/train_max
train_scaled <- as.data.frame(scale(digits_X,scale=FALSE,center=FALSE))
test_scaled <- as.data.frame(scale(test_$x/test_max,scale=FALSE,center=FALSE))


#collect the train/test labels for verification
train_labels <- factor(train_$y)
test_labels <- factor(test_$y)


############## run PCA
pca_train_ <- prcomp(as.matrix(train_scaled))
plot(pca_train_,type="lines",main="Scree Plot")

# Check the variance explained by the dimensions 
# create dataframe with cumulative variance and dimensions
varianceExplained <- as.data.frame(pca_train_$sdev^2/sum(pca_train_$sdev^2))
varianceExplained = cbind(c(1:784),cumsum(varianceExplained))
colnames(varianceExplained) <- c("PCs","Var") 

# create plots for variance 
ggplot(varianceExplained, aes(PCs,Var)) + 
  geom_point(color="blue") + labs(x= "Number of Dimensions",y= " Cumulative Variance") + ggtitle("PCA Dimensionality Reduction on MNIST")

# 200 PC's = 96.6% of variance

############## Run PCA on  covariance matrix
digits_covMatrix <- cov(train_scaled) 

#run PCA with Covariance Matrix
pca_train <- prcomp(digits_covMatrix,scale=FALSE,center=FALSE)
plot(pca_train,type="lines",main="Scree Plot (Covariance Matrix)")

# Check the variance explained by the dimensions 
# create dataframe with cumulative variance and dimensions
varianceExplained <- as.data.frame(pca_train$sdev^2/sum(pca_train$sdev^2))
varianceExplained = cbind(c(1:784),cumsum(varianceExplained))
colnames(varianceExplained) <- c("PCs","Var") 

# create plots for variance 
ggplot(varianceExplained, aes(PCs,Var)) + 
  geom_point(color="blue") + labs(x= "Number of Dimensions",y= " Cumulative Variance") + ggtitle("PCA Dimensionality Reduction (Covariance Matrix) on MNIST")

ggplot(varianceExplained, aes(PCs,Var)) + 
  geom_point(color="blue") + labs(x= "Number of Dimensions",y= " Cumulative Variance") + ggtitle("PCA Dimensionality Reduction (Covariance Matrix) on MNIST Zoomed")+
  xlim(c(0,40))

# 30 PC's == 97.7% of variance 
######
## Use the covariance matrix since it accounts for more variance using significantly less PC's 
######

# Do the dimensional reduction of the matrix to 30 columns for Covariance Matrix
rotate <- pca_train$rotation[,1:30]
trainFinal <- data.frame(as.matrix(train_scaled) %*% (rotate))
testFinal <- as.matrix(test_scaled) %*% (rotate)
trainFinal <- data.frame(trainFinal)
testFinal <- data.frame(testFinal)

# Plot of rotated data
ggplot(trainFinal, aes(PC1,PC2,colour=factor(train_$y))) + geom_point() +
  labs(x= "PC1",y= "PC2") + ggtitle("Rotated PC's Plot")



########## Reconstruction of MNIST data
xhat <- t(t(pca_train_$x[,1:30] %*% t(pca_train_$rotation[,1:30])) + pca_train$center) 

# show original digit
show_digit(train_$x[1,])
show_digit(train_$x[2,])

# show digit after PCA analysis
show_digit(xhat[1,])
show_digit(xhat[2,])



######################################################
# KNN 
######################################################
library(class)
set.seed(123)

# run KNN for PC's with K varying from 1 through 20 with PCA parameters
knnAcc <- matrix(nrow=10,ncol=2)
i=1
ptm <- proc.time()
for (k in 1:10){
  prediction = knn(trainFinal,testFinal,train_labels,k=k)
  knnAcc[i,] <- c(k,mean(prediction==test_labels))
  i = i+1
}
elapsed <- proc.time() - ptm
print(paste0(elapsed[3] , " seconds elapsed")) 

# convert accuracy to dataframe to be plotted
colnames(knnAcc) <- c("K","Accuracy")
knnAcc <- data.frame(knnAcc)

# plot KNN accuracy and determine the correct number for K
ggplot(knnAcc, aes(K,Accuracy)) + geom_line(color="red") + geom_point(color="red") +
  ggtitle("KNN - Accuracy by K-value")

# best prediction is at k=5
ptm <- proc.time()
best_prediction = knn(trainFinal,testFinal,train_labels,k=5)
elapsed <- proc.time() - ptm
print(paste0(elapsed[3] , " seconds elapsed")) 

#calculate the accuracy for k=5
hit = 0
for (i in 1:10){
  hit = hit + table(best_prediction,test_labels)[i,i]
}
table_prediction <- table(prediction,test_labels)
accuracy = hit/sum(table_prediction)
print(accuracy)

library(gridExtra)
library(grid)
grid.table(table_prediction)


# Run KNN with entire dataset parameters (all 784) NO PCA
# Use k=27 since sqrt(784) = 28 and rounded to nearest odd number
ptm <- proc.time()

prediction = knn(train_scaled,test_scaled,train_labels,k=27)

# print accuracy
hit = 0
for (i in 1:10){
  hit = hit + table(prediction,test_labels)[i,i]
}
table_prediction <- table(prediction,test_labels)
accuracy = hit/sum(table_prediction)
print(accuracy)

elapsed <- proc.time() - ptm
print(paste0(elapsed[3] , " seconds elapsed")) 




######################################################
# Random Forest
######################################################
library(randomForest)
library(readr)
library(caret)

set.seed(111)
numTrees <- 60

# run random forest with PCA reduced parameters
ptm <- proc.time()
rf <- randomForest(trainFinal,train_labels,ntree=numTrees)
plot(rf)
pred <- predict(rf,testFinal)
acc <- confusionMatrix(pred,test_labels)
acc$overall[1]
elapsed <- proc.time() - ptm
print(paste0(elapsed[3] , " seconds elapsed")) 

# Bagging 
set.seed(112)
rfAcc <- matrix(nrow=9,ncol=2)

ptm <- proc.time()
r = 1
c = 1
for (i in 2:10){
  bag <- randomForest(trainFinal,train_labels,ntree=numTrees,mtry=i) 
  pred <- predict(bag,testFinal)
  acc <- confusionMatrix(pred,test_labels)
  rfAcc[r,] <- c(i,acc$overall[1])
  r = r+1
  c = c+1
}
elapsed <- proc.time() - ptm
print(paste0(elapsed[3] , " seconds elapsed")) 

# determine optimal number of variables available for splitting at each tree node
plot(rfAcc,main="Optimal number of variables at each node",xlab="Number of variables",ylab="Accuracy")

# random forest for mtry = 4
optimal_bag <- randomForest(trainFinal,train_labels,ntree=300,mtry=4)
summary(optimal_bag)

pred <- predict(optimal_bag,testFinal)
acc <- confusionMatrix(pred,test_labels)
acc$overall[1]

# plot bag statistics
barplot(optimal_bag$err.rate[60,2:11]*100,main= "Error Rate By Class",xlab="Class",ylab = "Error Rate (%)")

######################################################
# Boosting 
######################################################
library(adabag)

ptm <- proc.time()
adaboost_df <- data.frame(trainFinal,train_labels)
ab <- boosting(train_labels~.,adaboost_df,boos=TRUE,mfinal=60)
elapsed <- proc.time() - ptm
print(paste0(elapsed[3] , " seconds elapsed")) 
pred <- predict(ab,testFinal)
acc <- confusionMatrix(factor(pred$class),test_labels)
acc$overall[1]
elapsed <- proc.time() - ptm
print(paste0(elapsed[3] , " seconds elapsed")) 


# run Random Forest NO PCA
ptm <- proc.time()
# 27 chosen as sqrt of 784 
rf <- randomForest(train_scaled,train_labels,ntree=numTrees)
plot(rf)
pred <- predict(rf,test_scaled)
acc <- confusionMatrix(pred,test_labels)
acc$overall[1]
elapsed <- proc.time() - ptm
print(paste0(elapsed[3] , " seconds elapsed")) 

library(gbm)
outGbm <- gbm.fit(train_scaled,  train_labels, distribution="multinomial",n.trees=500, interaction.depth=6)
pred <- predict(outGbm,trainFinal)
acc <- confusionMatrix(pred,test_labels)

predGbm <- factor(apply(predict(outGbm, testFinal, n.trees=outGbm$n.trees),1,which.max) - 1L)
acc <- confusionMatrix(predGbm,test_labels)
acc$overall[1]


