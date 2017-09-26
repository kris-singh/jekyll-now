---
layout: post
title:  Adaptive Sampling for SGD by Exploiting Side Information
tags: sgd variance reduction
---

## Introduction:
Your thesis is only a few days away. You have to read all the relevant papers and make a summary of them as quickly as possible.
But the because of the latest season of narcos and got coming back to back you are severly short of time. You realise that you can 
put your machine learning skills to use and classify all the relevant and non relevant papers by creating a model. You decide to use TF-IDF
frequency vector for training your NN with vanilla-SGD algorithm since you love the vanilla flavoured ice-cream and want are craving it too much under the stress. Here starts a research journey through building his experiment.

The reasearcher starts by first finding out the time required for SGD to converge. He knows SGD has a convergence rate of around O(1/k) for strongly convex loss function and around O(1/sqrt(k)) for convex functions. He also knows for having good enough classifier he would require the model loss functions should be around 10e3. Using some simple calculation he finds that it would require around 3 hours to train model on strongly convex function.
Since he is a little rusty on proving if the the loss function is strongly convex or not. So he overestimates the max time to 6 hours. He is impatient and can't wait this long for the results as his submission is due in a few days. He thinks to himself what can i do to make sgd faster.....

## Bias/Variance Tradeoff
Bias: It is the diffrence between the expected value of your model and real expected value. Expectations are always calculate with respect to a probability distribution. Here the probability lies in stochasticness of the sample points given to you. If you train your model again and again on diffrent samples of data. Your expected results should vary from the actual results as little as possible. Mathematically this can be represented as 
B[f(x)] = E[f(x) - f(x)] = E(f'(x)) - E(f(x)) [LOE]

Experiment:
```

```

Variance: If you allow to trained many times using diffrent dataset. The diffrence in prediction values for any given data point is called variance.
For more information and very good isllustration you can visit scoot great article.


### SGD Bias and Variance:
Okay lets first look at the Gradient Descent Optimisation. Given a loss function L and set of training points gradient descent finds a optimum point that gives the lowest error for the given loss function. It does so by moving in the direction of gradient at each iteration.
x+ = x - \sum_i=0^n \grad(L(x_i))
Now rember that this is a very expensive process since the gradient is summed up for all the data points. But also rember that this is the true gradient of the loss function.
Now suppose instead of choosing all data points i estimate the gradient by only a small set of data points of size 'm'. This is known as mini-batch SGD.
A more extreme case of this is to m = 1 this particular case is known as SGD. Now rember that we talked about the Bias and Variance when estimation quantities this also holds true for SGD. It can be shown that the SGD if run for long enough would give me an unbiased estimate of the Gradient. This simply means that the in expectaion i am actually moving along the correct direction and that turns out to be good enough. Though the problem here lies with variance of the estimates. Every gradient evaluated at a single point can give me very diffrent point and i might taking directions that actually taking me away from the minima. This is the reason for the poor convergence of SGD.

### Existing Solutions.
1. Averaging: One of the easiest solution is to use a minibatch. This can be thought of as avergaing the gradients and then taking a direction of descent.
2. Momentum: Nestrov Momentum, Adam, Adagrad actually use information from previous graients for correction of direction of gradient.
3. Tweaking Learning rates: You can tweak the learning rates so that they are inversely propotinal to the magnitude of the gradient. Hence never taking a big enough jump and reducing the variance.

### Adaptive Sampling Methods
The paper uses a known technique in Active Learning and RL called importance sampling. The basic concept of importance sampling is to add to weights to samples from data points. The weight scheme is in general static and reprsents a sort of confidence of the experimenter on the data points for giving good approximation to the gradient.
