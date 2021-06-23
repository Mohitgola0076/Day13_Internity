'''
                    # Python | Introduction to Matplotlib       
                    
Matplotlib is an amazing visualization library in Python for 2D plots of arrays. Matplotlib is a multi-platform data visualization library built on NumPy arrays and designed to work with the broader SciPy stack. It was introduced by John Hunter in the year 2002.

One of the greatest benefits of visualization is that it allows us visual access to huge amounts of data in easily digestible visuals. Matplotlib consists of several plots like line, bar, scatter, histogram etc.

        Installation :

Windows, Linux and macOS distributions have matplotlib and most of its dependencies as wheel packages. Run the following command to 

        Install matplotlib package :

python -mpip install -U matplotlib

        Importing matplotlib :

from matplotlib import pyplot as plt
or
import matplotlib.pyplot as plt 


                # What is a Histogram : 
                
A histogram is a plot of the frequency distribution of numeric array by splitting it to small equal-sized bins.

If you want to mathemetically split a given array to bins and frequencies, use the numpy histogram() method and pretty print it like below.

                #  Example : 
import numpy as np
x = np.random.randint(low=0, high=100, size=100)

# Compute frequency and bins
frequency, bins = np.histogram(x, bins=10, range=[0, 100])

# Pretty Print
for b, f in zip(bins[1:], frequency):
    print(round(b, 1), ' '.join(np.repeat('*', f)))
    
                # The output of above code looks like this:

10.0 * * * * * * * * *
20.0 * * * * * * * * * * * * *
30.0 * * * * * * * * *
40.0 * * * * * * * * * * * * * * *
50.0 * * * * * * * * *
60.0 * * * * * * * * *
70.0 * * * * * * * * * * * * * * * *
80.0 * * * * *
90.0 * * * * * * * * *
100.0 * * * * * *
 
 
                #  Plotting categorical variables  : 
                
You can pass categorical values (i.e. strings) directly as x- or y-values to many plotting functions:

                # Example : 
                
import matplotlib.pyplot as plt

data = {'apple': 10, 'orange': 15, 'lemon': 5, 'lime': 20}
names = list(data.keys())
values = list(data.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
axs[0].bar(names, values)
axs[1].scatter(names, values)
axs[2].plot(names, values)
fig.suptitle('Categorical Plotting')

                # Output : 
                https://matplotlib.org/stable/_images/sphx_glr_categorical_variables_001.png
                
                
                            #   Matrix plots in Seaborn
                            
Seaborn is a wonderful visualization library provided by python. It has several kinds of plots through which it provides the 
amazing visualization capabilities. Some of them include count plot, scatter plot, pair plots, regression plots, matrix plots and 
much more. This article deals with the matrix plots in seaborn.
       
                # Example : 
       
# import the necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt % matplotlib inline

# load the tips dataset
dataset = sns.load_dataset('tips')

# first five entries of the tips dataset
dataset.head()

# correlation between the different parameters
tc = dataset.corr()

# plot a heatmap of the correlated data
sns.heatmap(tc)

                
                #  Heatmap of the correlated matrix : 
                
Inorder to obatin a better visualisation with the heatmap, we can add the parameters such as annot, linewidth and line colour.

                    # Example : 

# import the necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt % matplotlib inline
  
# load the tips dataset
dataset = sns.load_dataset('tips')
  
# first five entries of the tips dataset
dataset.head()
  
# correlation between the different parameters
tc = dataset.corr()
sns.heatmap(tc, annot = True, cmap ='plasma', 
            linecolor ='black', linewidths = 1)

            # Add Grid Lines to a Plot  : 
            
With Pyplot, you can use the grid() function to add grid lines to the plot.
    
            # Example : 
Add grid lines to the plot:

import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid()

plt.show()


            
                # Specify Which Grid Lines to Display : 
                
You can use the axis parameter in the grid() function to specify which grid lines to display.

Legal values are: 'x', 'y', and 'both'. Default value is 'both'.

                 # Example : 
                 
Display only grid lines for the x-axis:

import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid(axis = 'x')

plt.show()


                #   matplotlib.pyplot.scatter() in Python : 
                
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is used for 
plotting various plots in Python like scatter plot, bar charts, pie charts, line plots, histograms, 3-D plots and many more. We 
will learn about the scatter plot from the matplotlib library.


                        # Syntax : 
The syntax for scatter() method is given below:

matplotlib.pyplot.scatter(x_axis_data, y_axis_data, s=None, c=None, marker=None, cmap=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None)


        # Example 1: This is the most basic example of a scatter plot.
        
import matplotlib.pyplot as plt

x =[5, 7, 8, 7, 2, 17, 2, 9,
	4, 11, 12, 9, 6]

y =[99, 86, 87, 88, 100, 86,
	103, 87, 94, 78, 77, 85, 86]

plt.scatter(x, y, c ="blue")

# To show the plot
plt.show()


                        # Style Plots using Matplotlib : 
                        

Matplotlib is the most popular package or library in Python which is used for data visualization. By using this library we can 
generate plots and figures, and can easily create raster and vector files without using any other GUIs. With matplotlib, we can 
style the plots like, an HTML webpage is styled by using CSS styles. We just need to import style package of matplotlib library. 


First, we will import the module:

from matplotlib import style

                    # Example : 

from matplotlib import style
print(plt.style.available)

Output:

[‘Solarize_Light2’, ‘_classic_test_patch’, ‘bmh’, ‘classic’, ‘dark_background’, ‘fast’, ‘fivethirtyeight’, 
‘ggplot’,’grayscale’,’seaborn’,’seaborn-bright’,’seaborn-colorblind’, ‘seaborn-dark’, ‘seaborn-dark-palette’, ‘seaborn-darkgrid’, 
‘seaborn-deep’, ‘seaborn-muted’, ‘seaborn-notebook’, ‘seaborn-paper’, ‘seaborn-pastel’, ‘seaborn-poster’,’seaborn-talk’,’seaborn-
ticks’,’seaborn-white’,’seaborn-whitegrid’,’tableau-colorblind10′]


        # Above is the list of styles available in package.

Syntax: plt.style.use(‘style_name”)



                    # Example 1:

# importing all the necessary packages
import numpy as np
import matplotlib.pyplot as plt

# importing the style package
from matplotlib import style

# creating an array of data for plot
data = np.random.randn(50)

# using the style for the plot
plt.style.use('Solarize_Light2')

# creating a plot
plt.plot(data)

# show plot
plt.show()
