# neural-networks
Implementation of Logistic Regressor, Shallow and Deep Neural Network from scratch

### Logistic Regression
Logistic Regression is a classification technique used to predict two distinct set of classes on the basis of given observations. Unlike linear regression where continious outputs values are generated, the sigmoid function in logistic regression converts those output values into a probability score which can be utilized to sperated the two distinct classes.

### Shallow Neural Network
In a shallow neural network, a neuron/node provides a certain output to next layer upon feeding an input from previous layer. The node has performs 2 tasks in between: first it calculates the sum of weights and in second part it feeds the sum into activation function to get an ouptut.
We build a regressor and neural network, and test with blobs and circles dataset.

We have been given the two distinct classes to classify: horse, dog. We have taken the batch1 for training our neural network. We extract out particular classes to distinguish  along with the observation featuers in order to create the trainig data. Further we have normalized the training and test data by diving it by 255; since the image pixels ranges in between from 0–255 and we want values ranging between 0 and 1. We have increased the number of neurons in hidden layer to 21.

### Deep Learning Enhancements 

In order to improve the performance of the model, we have implemented Stochastic Gradient Descent and L2 Regularization

#### Stochastic Gradient Descent:

Stochastic Gradeint Descent is an optimization algorithm which is used along with back-propagation, where we calculate the gradients for each observation while model development. In gradient descent if we are having a large dataset, there are a lot of redundant calculations before every update of parameters for observations which are alike. Stochastic Gradient Descent does one update at a time and it faciliates more fluctuations to find better local minima. Also, in order to avoid overshooting, we generally reduce the learning rate.

#### L2 Regularization:

While model building in neural networks, training with larger epochs the weights become specialized(larger) which tends to reduce the performance of model. Models which are complex, and there is overfitting due to model development picking up noise in the data, we can use regularization to reduce the complexity and avoid overfitting. Using L2 regularization, the weights parameters reduce gradually towards 0, but exactly 0. We keep a regularization parameter called lambda which is a hyperparameter used for optimzation. 

