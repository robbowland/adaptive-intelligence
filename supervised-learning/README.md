# COM3240 Assignment 1: Supervised Learning on Handwritten Letters  
  
This repository contains code for running supervised learning on the **EMNIST** data set using *stochastic mini-batch gradient descent*. The code utilises *matrix operations* to improve performance.  
It contains *object oriented* code for defining a variable depth neural network, training it and testing it whilst optionally collection   
various stats.  
  
## Files & Folders

`[neuralnet]` - package containing neural network, activation function and weight initialisation implementation code.


`[data]` - contains dataset for running supervised learning. The contained `[experiments]` folder is where outputs from experiments with long run-times are stored.


`main.py` - entry point for running experiments. Contains various subsections for running experiments to satisfy each of the requirement of the assignment.

## Usage
To run an experiment, enter `main.py` and configure it to run the desired experiment (instructions contained within). Configure the parameters for the experiment in the relevant section in `main.py`. Run `main.py` (python3.8 was used when writing this) and ensure the relevant packages are installed (using anaconda should mitigate the need to install any).

## Mark Breakdown

| Section        | Mark        | Comments                                                     |
| -------------- | ----------- | ------------------------------------------------------------ |
| *Results & Discussion* | ***60.5/70*** |  **Q1 & Q2** - Good answer. The student should describe the problem of vanishing gradients in the context of error functions. The updating equations are not reported explicitly. The report is well written and the language adopted is technical.<br />**Q3** - Graph shows convergence, and point of convergence correctly identified. Limited discussion around this.<br />**Q4** - Accuracies reported considerably lower than control found in source. No discussion as to why this may be<br />**Q5** - Correct, detailed derivation. Code implemented correctly.<br />**Q6** - good job!<br />**Q7** - good job!<br />**Q8** - Nice figure and fair discussion. Some comparison to the linear model performance in the EMNIST paper should be made for full marks.|
| *Scientific Presentation & Code Documentation* | ***??/20*** |  Well-structured report and clear figures. Well done! Good suggestions to improving the model.                                               |
| *Modelling Originality* | ***??/10*** |             -                                      |

***Overall:    82.5/100   |   82.5%***
