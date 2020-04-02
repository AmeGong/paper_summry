# paper summary 
In recent years, artificial neural networks have shown excellent performance in pattern recognition and regression. (Deep learning in neural networks: An overview
) The convolutional neural network got unbelievable scores in image classification. So what is neural newtwork? Why doee it perform so good?

Let's begin by first understanding what consist of neural network. Neuron and activation function are the two main part in Neural network. In fact, A neuron is just a function, receiving several inputs and giving the result. Typically, those inputs are summed with weights, and this sum is passed through a nonlinear function, often called activation function, such as the sigmoid, relu, and so on. Above is a neuron model, and aritifical neural network consists of numbers of neurons. Different architecture of neurons have different performance. The fully connected network is the classic architecture, shown below.
![](./pic/fcn.png)
Neurons are placed in some layers, and each neuron in above layer is connected to all the neurons in below layer. The model is the basic architecture, having nice performace in some nonlinear problem. Other architectures is based on the fully connected neural network, and some architectures are significantlly effective in special domain, such as Convolution neuron network processing image, and Long short-term memory processing sequances of data. We are devoted to find more effective architectures to sovle different problems. Our goal is to find a architecture having excellent in different realms.

Now we know tha neural network is the combination of linear functions and nonlinear functions. So the challenge is how to determine the weights in the linear functions. We can denote neural network as y = NN(x). x is the inputs, and y is the outputs of the neural networks. In order to evaluate the NN function, we can define the loss(x,w) function according to different problem. So the challenge become a optimal problem to minimize the loss function. Intuitively we want to get the gradient of w. And the back propagation algorithm, a main algorithm in neural network, can get all parameters' gradient in a short time. 

Batch gradient descent, stochastic graident descent and mini-batch gradient descent are three different variants of gradient descent, which differ in how much data is used to computer the gradient of the loss function. Depending on the amount of data, the accuracy and the training time will have an update. Batch gradient descent computes the gradient to update the parameters once using the entire data. Batch gradient can be very slow, as we need to load the whole data in memory and compute the gradient for one update. In contrast, stochastic gradient descent execute a paramter update for each training data. stochastic gradient descent performs much faster than batch gradient descent. But one training data might contain too much noise leading to bad result. The mini-batch gradient descent perform well in both time and effect. The mini-batch gradient descent perform a parameter update for every mini-batch of n training samples, So mini-batch gradient descent overcome the problems of time and effect.

But a another challenge is to minimize the non-convex error function with high demension. Gradient descent will be trapped in the local minmum to have a bad performance. In order to  overcome this, some variants can be applied to solve the question. Momentum can help accelerate stochastic graident descent in the relevant direction, and it does this by adding a fration r of the update vector of the past time step to the current update vector.
![](./pic/momentum.png)

Nesterov accelerated gradient is a way to give the next position and recompute the present gradient using the next position to update the parameters.
![](./pic/nag.png)

Those above method all have a drawback that all parameters update themselves in a same learning rate. Adagrad is a method to adapt the learning rate to different paramters, having a great progress. Briefly Adagrad modifies the gerenal learning rate for each parameter based on the past gradient.
![](./pic/adagrad.png)

Adadelta is an extension of Adagrad that seek to solve the monotonically decreasing learning rate. Instead of accumulating all past squared gradient, Adadelta uses an exponentially decaying average to prevent the diminishing learning rates.
![](./pic/adadelta.png)

Adam method is variant to sovle the diminshing learning rate in Adagrad. Adam as well store an exponentially decaying average of past squared gradients and gradient to adjust the learning rate. 
![](./pic/adam1.png)

In order to get the bias-corrected first and second moment estimates:
![](./pic/adam2.png)

Then we use those moments to update the parameters just as we have seen in Adadelta.
![](./pic/adam3.png)

Nadam method is the combination of Adam and Nesterov accelerated gradient.

Following is the visualization of those algorithms.
![](./pic/visualization.png)

So, which optimizer should we use? If the input data is sparse, adaptive learning-rate methods are most likely to achieve the best results.

Opimization can cause another problem,overfitting, which means that the model performs very well in training samples, but it has a bad accuracy in the test samples. many of the complicated relastionships in the train samples will be the result of sampling noise. There are many mehtods to reducing it. Those include stopping training as soon as performance start to get worse, introducing weight penalties of various kinds such as L1 and L2 regularization and drop out. Before drop out, model combination should be introduced. Model combination nearly always improves the performance of machine learning. Model combination is to average different models' outputs. But differnet model need either different architectures or different training samples. Training different architectures models is hard because the optimization of the model is a hard task and training different models need large computation. Moreover, large network require large amount of training data.With the limitation of computation and data, we can't train different models to combine different models. In order to 