# regionalVariance

## You can go directly to graphVariances.ipynb to inspect the graphs related to Goldenson et al., 2017 (submitted).

To generate the data from raw climate model output you can follow these steps:

### 0. Set directory paths to the climate model output files in constants.py

### 1. Spatial average to generate time-series, store for further use.

There are two versions of the script that takes monthly mean climate model outputs for some field and turns them into regional averages for whatever lat-lon boxes you choose to define.
	- [first one] is for CMIP5 formatted output
	- [second one] is for CESM LE output
Slight adaptations might be needed for different model output formats.

In case you don’t have all of the raw model output available on your machine, the results of these scripts for the regions in Goldenson et al., 2017 are already in the subdirectories specified in constants.py. They are .csv files saved-out using the Python library Pandas.

### 2. Further process and convert time-series data into R dataframe format and save.
Now that you have generated and saved regional average time series for the regions you have defined and named, and for the climate variables of choice, you should also create a version of these that is in the form of a R dataframe because some of the analysis will require the use of R. 
	- run script makeRdataFrame.py

### 3. run R script
I recommend R Studio if you are using a Macintosh.

You must obtain the code from Paul Northrup that accompanies Northrup and Chandler (2014) at  http://www.homepages.ucl.ac.uk/~ucakpjn/SOFTWARE/NorthropChandler2014.zip

Unpack it as a sub-directory of the main project directory where you have your copy of this. Copy or move appliedExample.R, and appliedExampleBayes.R into the same directory.   

appliedExample.R is the one that is run to get the results in Goldenson et al., 2017. you will want to edit it to loop over the climate variables, regions, and other variants that you prefer. Hand edit the lines to set the working directory and other directory paths, and to specify whether or not to calculate confidence intervals. You might not want to run this for a lot of variations sequentially with confidence intervals because it will take a while.

The output of the R script should up ends up in the subdirectory NCresults off of this main directory. Sample data is already there.

### 4. run graph-making notebook
Now all of the files are present to run the plot-making scripts found in the Jupyter Notebook graphVariances.ipynb See descriptions therein. Examine class defaults in uncertaintyFunctions.py to understand all of the options that can be set if it is not clear enough from the examples.



