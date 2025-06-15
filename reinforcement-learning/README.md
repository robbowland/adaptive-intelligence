# COM3240 Assignment 2: Reinforcement Learning

This repository contains code for running reinforcement learning tasks within a 2-dimensional gridworld environment.

## Files & Folders
`[agent]` - contains code defining functionality related to agent behaviour, such as the agent itself and its policy.

`[environment]` -  contains code related to the simulation environment, including the environment itself and actions that are permitted for usage within the environment.

`main.py` - entry point for running experiments, contains various configurables for altering experiment parameters.

## Usage
Alter the desired parameters and run `main.py` (python3.8 was used when writing this) and ensure the relevant packages are installed (using anaconda should mitigate the need to install any).

## Mark Breakdown

| Section        | Mark        | Comments                                                     |
| -------------- | ----------- | ------------------------------------------------------------ |
| *Results & Discussion* | ***55/70*** | **Q1** - Good job<br/>**Q2** - Graphs provided are clear, would have been nice to have them overlayed for easier comparison, but doesn't limit comparability. Discussion brief but accurate, with points made evidenced in the graphs provided. Would have been useful to have purely greedy instead of low epsilon value for better comparison.<br/>**Q3** - Graphs show clear parameter optimistion, discussion around the observed trends is good for backing up parameter selection and grounding findings in theory.<br/>**Q4** - well done, great explanation.<br/>**Q5** - Good idea to show the Q-value magnitude as well as direction but no plot for no obstacle version.<br/>**Q6 & Q7** - The solution to the problem of high dimensional space is correct, but the student could have considered also to adopt a continuous representation of the environment (space cells). The trade-off between exploration and exploitation could have been described in the case where walls are present. |
| *Scientific Presentation & Code Documentation* | ***??/20*** |  The report overall has a good layout and neat figures. It should be of journal standard for higher marks.                                               |
| *Modelling Originality* | ***??/10*** |             -                                      |

***Overall:    72.5/100   |   72.5%***
