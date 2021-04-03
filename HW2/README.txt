I am submitting 3 .py files along with report and dataset as requested: 

1- knn.py : knn implementation. 
	knn(train_data,train_labels,test_data,k): is the main knn algorithm. It takes training data and test data and applies the knn algorithm with parameter k. it returns the predictions of the test data.  
	ten_fold_cross_knn(..) : I implemented the 10 fold cross validation, it returns the accuracy list obtained from the 10 training/validation. 
	train(...) : it is the requested algorithm in the homework. It applies 10-fold cross validaton with k = 1,3,5,....,199
	test(...) : it applies knn algorithm to give test data using given train data with k parameter.

	I have called the train algorithm and get the best resulting k and applied test  function in the file. İt will give the output requested in the hw. 

2- kmeans.py
	main algorithm is kmeans(data,k,plot,eps). if plot is true, it plots every iteration. if not, plots only the result.
	plot_elbow : plots the requested objective functiıon - k  plots. 

	I have called plotting elbows and kmean algorithm with the chosen k's. It outputs the requested output in the hw. 

3- hac.py

	hac(data , k , criterion ) : is the main algortihm. it takes criterion as input and choses the distance function according to that.  It merges clusters up to k.

	I called all the hac(data,k,crit) at the end, it will plot the last 10 steps of each clustering. It is also printing total number of clusters so that I could keep track of.