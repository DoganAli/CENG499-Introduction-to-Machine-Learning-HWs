There are 4 .py files, a directory logs that contains the logs of trainings and accuracies and one report.pdf

dataset.py : It contains the dataset module that is derived from the videos provided.

model.py : Contains the fully connected NET model that is requested and used in the homework

cnn.py : Contains a CNN model I have implemented to get better results.

train.py : Contains functions and model configurations such as train(..), accuracy_test(..), hyperparameter_tune(...). I used train.py to intiialize the models, optimizers, transforms and all for the training and testing.

Training :
  To train the model, we can use the train(..) function in train.py. However, before using hte function you have to create your model, specify optimizer, loss function, split the data into train and validation and pass the data loader. I have writen a function train_run_example() that calls the best result I had from trainings, also calls accuracy test and test() functions so that it will prepare labels.txt file for submission.
  There is an accuracy_test function that takes the model and data loaders and calculates accuracies of training and validation. I have called that function in train_run_example() as well.
  I used hyper_parameter_tune(..) function to tune the hyper parameteres. You can find the related information inside the function.

Testing :
   To test the model on test data, you just need to call test(model) function and it will create a labels.txt file in the current directory.
