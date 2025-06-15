"""
Contains code for plotting data for experiments with a long running time where data was stored.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Plotting scatter with error bars
# https://stackoverflow.com/questions/22364565/python-pylab-scatter-plot-error-bars-the-error-on-each-point-is-unique

# Some formatting info
# https://stackoverflow.com/questions/24547047/how-to-make-matplotlib-graphs-look-professionally-done-like-this
# https://www.earthdatascience.org/courses/scientists-guide-to-plotting-data-in-python/plot-with-matplotlib/introduction-to-matplotlib-plots/customize-plot-colors-labels-matplotlib/

# line graph stuff
# https://stackoverflow.com/questions/20130227/matplotlib-connect-scatterplot-points-with-line-python

# font sizes
# https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib?rq=1

# Connected Scatter
# https://www.delftstack.com/howto/matplotlib/how-to-connect-scatterplot-points-with-line-in-matplotlib/

if __name__ == '__main__':
    Q3 = False
    Q6 = False
    Q7 = True
    Q8 = False
    if Q3:
        data = pd.read_csv(os.path.join('.', 'data', 'experiments', 'q3.csv'))

        ax = plt.subplot(111)
        # *-1 to get them to be the right way round, need to add on 0
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # To make them evenly spaced, I'll add the numbers in power point
        ax.plot(data['0.05'], '--', color="black")
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Axes Label & Title
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Weight Update', fontsize=12)
        plt.show()

    if Q6:
        data = pd.read_csv(os.path.join('.', 'data', 'experiments', 'q6_nnh50_e400_b50_lr0.05_r5.csv'))

        ax = plt.subplot(111)
        # *-1 to get them to be the right way round, need to add on 0
        x = [1,2,3,4,5,6,7,8,9,10,11]  # To make them evenly spaced, I'll add the numbers in power point
        ax.plot(x, data['Average Error'], '--o', color="black")
        ax.errorbar(x, data['Average Error'], yerr=data['Standard Deviation'],
                    lw=0.5,
                    capsize=2, capthick=1,
                    color="black")
        plt.xticks(x, ['0', '0.3', '0.03', '3e-3', '3e-4', '3e-5', '3e-6', '3e-7', '3e-8', '3e-9', '3e-10'])
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Axes Label & Title
        plt.xlabel('L1 λ Value', fontsize=12)
        plt.ylabel('Validation Error', fontsize=12)
        plt.show()

    if Q7:
        # Q7
        # Probably quicker to do titles and stuff in power point
        data = pd.read_csv(os.path.join('.','data','experiments', 'Q7.csv'))

        ax = plt.subplot(111)
        ax.plot(data['Hidden Layer Neurons'], data['Average Accuracy']*100, '--o', color="black")
        ax.errorbar(data['Hidden Layer Neurons'], data['Average Accuracy']*100, yerr=data['Standard Deviation']*100, lw=0.5,
                    capsize=2, capthick=1,
                    color="black")
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Axes Label & Title
        plt.xlabel('Number of Hidden Layer Neurons', fontsize=12)
        plt.ylabel('Test Data Classification Accuracy (%)', fontsize=12)
        plt.show()

    if Q8:
        # Q7
        data = pd.read_csv(os.path.join('.', 'data', 'experiments', 'q8_e400_b50_lr0.05.csv'))

        ax = plt.subplot(111)
        ax.plot(data['Hidden Layer Neurons'], data['Average Accuracy'] * 100, '--o', color="black")
        ax.errorbar(data['Hidden Layer Neurons'], data['Average Accuracy'] * 100,
                    yerr=data['Standard Deviation'] * 100, lw=0.5,
                    capsize=2, capthick=1,
                    color="black")
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Axes Label & Title
        plt.xlabel('Number of Neurons in Second Hidden Layer', fontsize=12)
        plt.ylabel('Test Data Classification Accuracy (%)', fontsize=12)
        plt.show()
