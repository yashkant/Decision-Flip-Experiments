# Decision-Flip-Experiments


To begin with experiments you will need Tensorflow and matplotlib installed on the system. CIFAR folder contains all the experiments performed in Jupyter notebooks.

**Helper Modules:**

-helper.py : Contains helper functions.  
-plotter.py: Contains graph plotting functions.  
-saveloader.py: Contains saving and loading functions.  


**The last cell of the notebooks executes the corresponding experiment, where the arguments could be as follows:**

*method = 2 (default setup, no need to tweak)*  
*label = "cifar_with-cnn" (decides which pre-trained model to load, fixed as of now!)*  
*n = 100 (No. of examples to be flipped per train, test, random, random-normal data, can be tweaked)*  
*epochs = 200 (No. of epochs for the pre-trained model, can be tweaked in multiples of 5,most of the notebooks use 100/200 )*  
*cls = -1 (perform per class experiment, pass -1 for all classes, not need to tweak*  
  
  
  
**Experiment Name : Vanilla Decision Flip Experiment ([vanilla-decision-flip.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/vanilla-decision-flip.ipynb))**

-Flips the given examples to all the possible classes for train, test, random, random-normal data.  
-Performs per class graphical analysis of the distances for the above mentioned types of data. Currently for classes 0,1.  

Todo: Add the measure for multiple boundary hits, prepare functions to save the plots and put it in plotter.   


**Experiment Name : Train again and Flip Experiment ([train_again.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/train_again.ipynb) and [train_again-cls_scores.ipynb](https://github.com/yashkant/Decision-Flip-Experiments/blob/master/CIFAR/train_again-cls_scores.ipynb) )**



