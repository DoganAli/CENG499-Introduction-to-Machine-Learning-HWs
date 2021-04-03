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


log_x_layers:
 The directories contain the optimization results. For example, log_3_layers contains three files, each contains 72 rows of train-cal accuries of the 3 layer model with an activation function specified on the directory name. They are created using save(..) function in train.py. I could not add every result to the report, that is why I put these logs. I also plotted every training, but due to its size, I did not add to submission but all can be accesed via : https://drive.google.com/drive/folders/1AdWtU8uy0mVOxEHF600j0wXIiNbfNhQj?usp=sharing
