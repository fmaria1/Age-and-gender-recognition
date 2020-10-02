# Age-and-gender-recognition

Facial image analysis is an important task for the Computer Vision but also a challenge for age and gender recognition, mainly due to large variability of resolution, deformation and occlusion.


The main objectives of the paper are the implementation of neural network architectures that manage to provide the most real and accurate results for problems such as age or gender prediction.


The topics follow the concept of image classification, in which the most important role is played by convolutional neural networks.
They are able to analyze the features detached from the images and learn them, and based on them to assign for each picture a label of belonging to a given class of the problem.
For the age recognition issue, the network returns a single value, namely age.
Due to the lack of resources, instead of analyzing the 1-100 year interval, the solution was minimized at the 18-50 year interval, being an interest interval for most fields of application.
In the case of the problem of gender prediction, the images are classified into two categories: female and male.

There are four main steps in solving these types of problems, namely:
-choosing an appropriate neural network architecture for the requirements of the problem;
-construction of a specific data set for each solution
-training, in which image features are extracted, and at the end of this step, the network is able to make predictions for new data.
-Finally, the network is tested to find out how accurate the predictions are and to analyze the results.
All implementations were made using the Python programming language, with the help of the PyTorch library specific to neural networks.


CNN are very similar to classical neural networks, but they only receive images as input data.
These by the type of layers used, their arrangement and parameters used are able to solve various recognition problems.
The first and best known architecture used for such problems is VGG16.
The ResNet50 architecture was also used to perform a comparative analysis and out of the desire to obtain the best possible results.
Both architectures are very well built, and their implementation has a fixed structure, except for some parameters that must be changed depending on the problem studied: for example the size of the input images or the number of classes.


In this training process all the images in the data set are passed through the created convolutional architecture and are analyzed.
Network training consists of two phases: forward and backward propagation and is performed over a set number of epochs appropriate to the complexity of the problem and the size of the data set used.
In the first phase, all the features extracted from each photo are propagated on the network, which teaches them.
The network is then able to provide a result, which is to be compared with the actual label of the original image.
The network losses are represented by the difference between the two labels, and mean square error was used as a cost function for their calculation.
The losses are propagated back to the network to review the information, using the Adam optimization algorithm.

Following the training of each neural network architecture, a model is obtained, and the next step in solving the problem is to test it.
The testing is performed using a new set of test data, which will compare the predictions returned by the trained model with the actual labels, and thus it will be possible to analyze the correctness of the results.

In the end, 4 trained models were obtained: 2 models for age recognition and 2 for gender prediction (one for each architecture). For both age recognition and gender prediction, the VGG16 architecture obtained much better predictions in the testing phase.
For the solution of age recognition, being a regression problem, an error as small as possible is pursued. The correctness of the model lies in the value of the average error of the test set between the predicted and the actual label. For the VGG16 architecture the error is 5.83 years, and for ResNet50 7.56 years.
For the gender prediction solution, the correctness is given as accurately as possible. VGG16 also obtained an accuracy better than 65.12%, compared to ResNet50 which obtained only 59.39%.
