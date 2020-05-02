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
mnist <- keras::dataset_mnist()
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

# input shape of the images (16x16)
input_shape <- c(16, 16, 1)

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


# set up convolutional model and dense model
	#TODO convolutional model may not be correctly set up yet
	model_conv <- keras_model_sequential() %>%
		layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', 
		input_shape = input_shape) %>%
		layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu' ) %>%
		layer_max_pooling_2d(pool_size = c(2,2)) %>%
		layer_flatten() %>%
		layer_dense(units = 128, activation = 'relu') %>%
		layer_dense(units = num_classes, activation = 'softmax')
		
	model_conv %>%
		compile(
			loss = "loss_categorical_crossentropy",
			optimizer = optimizer_adadelta(),
			metrics = c('accuracy'))
			
	
		
	model_dense <- keras_model_sequential() %>%
		layer_flatten(input_shape = input_shape) %>%
		layer_dense(units = 270, activation = "relu") %>%
		layer_dense(units = 270, activation = "relu") %>%
		layer_dense(units = 128, activation = "relu") %>%
		layer_dense(units = 10, activation = "softmax")
	
	model_dense %>%
		compile(
			loss = "loss_categorical_crossentropy",
			optimizer = optimizer_adadelta(),
			metrics = c('accuracy'))
			
# create list to hold accuracy data at the end of the following loop
network.data.dt <- list()
			
# TODO: Use x_train/y_train to fit the two neural network models described above.
# Use at least 20 epochs with validation_split=0.2 (which splits the train data 
# into 20% validation, 80% subtrain).
for(fold in fold.dt)
{
	x_subtrain = fold[2]
	y_subtrain = fold[3]
	#train convolutional network
	# same model architecture as in mnist_ccn_keras R example, but with the input_shape changed to 
	# reflect the size of the zip.train images (16x16). There should be 315146 total parameters to learn
	# The number of hidden units in each layer is 784, 6272, 9216, 128, 10.
	model_conv %>%
		fit(
			x = x_subtrain, y = y_subtrain,
			epochs = 20, validation_split = 0.2,
			verbose = 2
			)

	
	#train dense network
	# fully connected (784, 270, 270, 128, 10) network. The size of this netwokwr is deliberately chosen
	# to have a similar number of parameters to learn: 321,098
	model_dense %>%
		fit(
			x = x_subtrain, y = y_sbutrain,
			epochs = 20, validation_split = 0.2,
			verbose = 2
			)


	# TODO:  Compute validation loss for each number of epochs, and define a
	# variable best_epochs which is the number of epochs that results in 
	# minimal validation loss.
	val_loss <-  
	best_epochs <- 100


	# TODO: Re-fit the model on the entire train set using best_epochs and validation_split=0.
	model_conv %>%
			fit(
				x = X.mat, y = y.vec,
				epochs = best_epochs, validation_split = 0,
				verbose = 2
				)

		
		#train dense network
		#fully connected (784, 270, 270, 128, 10) network. The size of this network is deliberately 	chosen
		# to have a similar number of parameters to learn: 321,098
		model_dense %>%
			fit(
				x = X.mat, y = y.vec,
				epochs = best_epochs, validation_split = 0,
				verbose = 2
				)

	# TODO: Finally use evaluate to compute the accuracy of the learned model on the test set.
	# (proportion correctly predicted labels in the test set)
	conv_score <- model_conv %>% evaluate(fold[4], fold[5], verbose = 0)
	dense_score <- model_dense %>% evaluate(fold[4], fold[5], verbose = 0)

	# TODO: Also compute the accuracy of the baseline model,
	# which always predicts the most frequent class label in the train data.
	

	# TODO: At the end of your for loop over fold IDs,
	# you should store the accuracy values, model names,
	# and fold IDs in a data structure (e.g. list of data tables) for analysis/plotting.
	network.data.dt[fold] <- [scores, foldID]

} #end of loop over fold IDs

# TODO: Finally, make a dotplot that shows all 15 test accuracy values.
# The Y axis should show the different models, and the X axis should show the test accuracy values.
# There should be three rows/y axis ticks (one for each model),
# and each model have five dots (one for each test fold ID).
# Make a comment in your report on your interpretation of the figure.
# Are the neural networks better than baseline? 
# Which of the two neural networks is more accurate?
ggplot()+
  geom_tile(aes(
    x=network.data.dt, y=model_conv, fill=intensity),
    data=zip.some.tall)+
  facet_wrap("observation")
  coord_equal()+
