# MNIST Classification
Simple project using dimensionality reduction and machine learning to classify handwritten digit images.

The MNIST dataset is a publicly available dataset that is very often used for simple classification algorithms. This is a short project that loads in the MNIST dataset, uses Principle Component Analysis (PCA) to reduce the dimensionality of the data, and determines a simple machine learning technique for modeling of the data. The techniques that are applied in this project are: KNN and Random Forest. 

# Running the script 
In order to run the script MNIST_main.R, first the 4 data files necessary for this project need to be downloaded and stored in an approriate folder. The file data_README.txt contains the instructions to do so. 

Once the data is stored under the "data" folder, then the working directory must be set in the script MNIST_main.R. This is to ensure that the file will look in the correct location for the data necessary for the script to run.

# Results

Principle component analysis (PCA) was performed on the MNIST dataset. Results suggest that ~2% of the features can accurately represent 98% of the variance.  

![alt text](https://github.com/ldicarlo1/MNIST_Classification/blob/main/images/Screen%20Shot%202021-10-01%20at%204.29.31%20PM.png)

PCA representation of the images:

![alt text](https://github.com/ldicarlo1/MNIST_Classification/blob/main/images/Screen%20Shot%202021-10-01%20at%204.29.50%20PM.png)

Random Forest number of trees, accuracy and error by class:

![alt text](https://github.com/ldicarlo1/MNIST_Classification/blob/main/images/Screen%20Shot%202021-10-01%20at%204.30.31%20PM.png)


