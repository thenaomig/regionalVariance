# regionalVariance
This project is about regional climate uncertainty quantification.

This is the code to accompany Goldenson et al., 2017 (submitted). It takes climate model output (temperature, precipitation, etc.), averages over some geographic region, and quantifies the relative variance in each source of uncertainty about future climate projections.

### Go directly to <strong>graphVariances.ipynb</strong> to inspect the graphs related to Goldenson et al., 2017 (submitted).
The examples there generate the plots in the paper and its supplement, along with some additional graphs. They primarily use the functions in <strong>uncertaintyFunctions.py</strong> to process data and make the graphs.

The sample data in the project should allow the notebook to run as is. If you'd like to reproduce it, or define your own regions or climate variables...

#### Then, to generate the data from raw climate model output you can follow these steps:

#### 0. Set directory paths to the climate model output files in <strong>constants.py</strong>

#### 1. Spatial average to generate time-series, store for further use.

There are two versions of the script that takes monthly mean climate model outputs for some field and turns them into regional averages for whatever lat-lon boxes you choose to define.
- <strong>makeTimeSeriesCMIP5.ipynb</strong> is for CMIP5 formatted GCM output
- <strong>makeTimeSeriesCESM_LE.ipynb</strong> is for CESM Large Ensemble output

Slight adaptations might be needed for different model output formats.

In case you don’t have all of the raw model output available on your machine, the results of these scripts for the regions in Goldenson et al., 2017 are already in the subdirectory /timeSeries, as specified in <strong>constants.py</strong>. They are .csv files saved-out using the Python library <a href='http://pandas.pydata.org'>Pandas</a>.

#### 2. Further process and convert time-series data to R dataframe format and save.
Now that you have generated and saved regional average time series for the regions you have defined and named, and for the climate variables of choice, you should also create a version of the data that is in the form of a R dataframe because some of the analysis will require the use of R. The script first calculates anomalies against a historical period for annual means, winter, or summer season, and then smooths with a decadal running mean.

run script:

	python makeRdataFrame.py
You may first want to edit it to loop over the region names that you have defined.

#### 3. run R script
I recommend <a href='https://www.rstudio.com/products/rstudio/#Desktop'>R Studio</a> if you are using a Macintosh.

You must obtain the code from Paul Northrop that accompanies Northrop and Chandler (2014) at:  http://www.homepages.ucl.ac.uk/~ucakpjn/SOFTWARE/NorthropChandler2014.zip

Place it in the main project directory and unzip to create a sub-directory. Copy or move appliedExample.R, and appliedExampleBayes.R into that subdirectory.   

<strong>appliedExample.R</strong> is the one that is run to get the results in Goldenson et al., 2017. You may want to also examine Northrop and Chandler's exampleCode.R on which it is based, and their README.pdf

- edit appliedExample.R to loop over the climate variables, regions, and other variants that you prefer.
- edit the lines to set the working directory and other directory paths
- specify whether or not to calculate confidence intervals, by making sure the correct two lines - with or without them - are uncommented. (You might not want to run confidence intervals for a lot of variations sequentially because it will take a while.)

The output of the R script should end up in the subdirectory NCresults/ of this main directory. Sample data is already there.

#### 4. run graph-making notebook
Now all of the files are present to run the plot-making scripts found in the Jupyter Notebook <strong>graphVariances.ipynb</strong>

See descriptions therein. Examine <em>class defaults</em> in <strong>uncertaintyFunctions.py</strong> to understand all of the options that can be set if it is not clear enough from the examples.

###### Finally, please contact me if you find any issues or are trying to use this and have any questions!
