# Install packages if they are not already installed
if(!require("data.table"))
{
	install.packages("data.table")
}
if(!require("ggplot2"))
{
  install.packages("ggplot2")
}
if(!require("tensorflow"))
{
  install.packages("tensorflow")
}
if(!require("keras"))
{
  install.packages("keras")
}
if(!require("R.utils")){
  install.packages("R.utils")
}

# Load the libaries to be used later
library(data.table)
library(ggplot2)
library(tensorflow)
library(keras)
library(R.utils)

# Load zip.train.gz
# Download zip.train data set to local directory, if it is not present
if(!file.exists("zip.train.gz"))
{
  download.file("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz", "zip.train.gz")
}

# Read zip.train data set
zip.train.dt <- data.table::fread("zip.train.gz")
N.obs <- nrow(zip.train.dt)
X.mat <- as.matrix(zip.train.dt[, -ncol(zip.train.dt), with=FALSE])
y.vec <- zip.train.dt[[ncol(zip.train.dt)]]

# For 5-fold cross-validation, create a variable fold_vec which randomly
# assigns each observation to a fold from 1 to 5.
set.seed(1337)
fold_vec <- rep(sample(1:5), l=nrow(X.mat))


# For each fold ID, you should create variables x_train, y_train, x_test, y_test 
# based on fold_vec.

# Initalize a list to hold folds and their respective test train
fold.dt.list <- list()

# Loop through folds to create fold specific data
for(test.fold in 1:5) {
  is.test <- fold_vec == test.fold
  is.train <- !is.test
  x_train <- X.mat[is.train,]
  y_train <- y.vec[is.train]
  x_test <- X.mat[is.test,]
  y_test <- y.vec[is.test]
  
  # Save generated fold specific data
  fold.dt.list[[test.fold]] <- data.table::data.table(
    test.fold,
    x_train,
    y_train,
    x_test,
    y_test
  )
}
fold.dt <- do.call(rbind, fold.dt.list)

# TODO: Use x_train/y_train to fit the two neural network models described above.
# Use at least 20 epochs with validation_split=0.2 (which splits the train data 
# into 20% validation, 80% subtrain).


# TODO:  Compute validation loss for each number of epochs, and define a
# variable best_epochs which is the number of epochs that results in 
# minimal validation loss.
best_epochs <- NULL


# TODO: Re-fit the model on the entire train set using best_epochs and validation_split=0.


# TODO: Finally use evaluate to compute the accuracy of the learned model on the test set.
# (proportion correctly predicted labels in the test set)


# TODO: Also compute the accuracy of the baseline model,
# which always predicts the most frequent class label in the train data.


# TODO: At the end of your for loop over fold IDs,
# you should store the accuracy values, model names,
# and fold IDs in a data structure (e.g. list of data tables) for analysis/plotting.


# TODO: Finally, make a dotplot that shows all 15 test accuracy values.
# The Y axis should show the different models, and the X axis should show the test accuracy values.
# There should be three rows/y axis ticks (one for each model),
# and each model have five dots (one for each test fold ID).
# Make a comment in your report on your interpretation of the figure.
# Are the neural networks better than baseline? 
# Which of the two neural networks is more accurate?