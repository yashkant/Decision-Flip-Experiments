# Decision-Flip-Experiments


To begin with experiments you will need Tensorflow and matplotlib installed on the system. CIFAR folder contains all the experiments performed in Jupyter notebooks.

**Helper Modules:**

- helper.py : Contains helper functions.  
- plotter.py: Contains graph plotting functions.  
- saveloader.py: Contains saving and loading functions.  
- fgsm*.py : Contain different versions of modified Fast Gradient Sign Method refer [here](https://arxiv.org/abs/1412.6572).


**The last cell of the notebooks executes the corresponding experiment, where the arguments could be as follows:**

*method = 2 (default setup, no need to tweak)*  
*label = "cifar_with-cnn" (decides which pre-trained model to load, fixed as of now!)*  
*n = 100 (No. of examples to be flipped per train, test, random, random-normal data, can be tweaked)*  
*epochs = 200 (No. of epochs for the pre-trained model, can be tweaked in multiples of 5,most of the notebooks use 100/200 )*  
*cls = -1 (perform per class experiment, pass -1 for all classes, not need to tweak*  
 
  
  
**Experiment Name : Vanilla Decision Flip Experiment ([vanilla-decision-flip.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/vanilla-decision-flip.ipynb) and [layerwise-experiment.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/layerwise-experiment.ipynb))**

- Flips the given examples to all the possible classes for train, test, random, random-normal data.  
- Performs per class graphical analysis of the distances for the above mentioned types of data. Currently for classes 0,1.  
-`layerwise-experiment.ipynb` compares the distances across each layer as well. 

Todo: Add the measure for multiple boundary hits, prepare functions to save the plots and put it in plotter.   


**Experiment Name : Train again and Flip Experiment ([train_again.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/train_again.ipynb) and [train_again-cls_scores.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/train_again-cls_scores.ipynb) )**

- Compares the distances across given examples on uniformly pre-trained model vs overfitted model on these examples.  
- File `train_again-cls_scores.ipynb` also displays the class scores along with the comparison plots


**Experiment Name : Multiple Hits Experiment ([multiple-hits-experiment-diff.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/multiple-hits-experiment-diff.ipynb) and [multiple-hits-experiment-e200.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/multiple-hits-experiment-e200.ipynb))**

- This experiment detects the points which hit any other class boundaries before flipping themselves to the target class. 
- File `multiple-hits-experiment-e200.ipynb` contains the code for a model trained to 200 epochs
- File `multiple-hits-experiment-diff.ipynb` compares the hits for 100 and 200 epochs models 




**Experiment Name : Layerwise Own to Flip ratio experiment ([layerwise-own-to-flip-ratio-experiment.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/layerwise-own-to-flip-ratio-experiment.ipynb))**

- Experiment to compare the distance of a point from its own class members and the nearest flipped point across different layers of the model. 


## Experiment Results 

**Decision Flip Experiments:** 


We find the derivative of Loss (Cost Function used to optimize the parameters of the model) w.r.t to the features (dimensions) of the image and scale its norm with a hypeparam (epsilon).

The quantity received is used as a perturbation until the label of the data point is flipped. So, in this case we end up taking steps of equal magnitude in the gradient’s direction. 

We compare the results across various initial and target classes and also across different layers : 



**Overfit and Flip Experiment:**

To more concretely analyze our results we overfit our model on a randomly chosen datapoint from the training set and then proceed to analyze the decision boundaries before and after overfitting. 

The results are as shown below :



Occasionally there were a few points for which the decision boundaries originally were very close and after overfitting were pushed  away, one of them is shown below: 

**Own Class to Flipped Ratio Experiment :**

Here we compare the distances of a point’s own class to the nearest flipped point using the above stated algorithm, we compare these distances across different layers of the model.


It is observed that as we go deeper and deeper in the model layers the distance from the members of own class decreases whereas the distance from flipped point increases and hence the ratio of  own/flip plotted across different layers of the model follows a decaying plot as follows: 








