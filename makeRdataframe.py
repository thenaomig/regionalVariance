# on the Macintosh, just type: 'python makeRdataframe.py'
# in the Terminal in this directory
import numpy as np
import scipy as sp
import pandas as pd
import os.path
import statsmodels.api as sm
from copy import deepcopy
import attr

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
saveRDS = ro.r('saveRDS')

from uncertaintyFunctions import *
from constants import *
    
def makeRdataframe(field,region,smoothed):
    '''make and store an R dataframe for just one year at a time
    by looping over year'''
    for year in years:
        oneTime = smoothed[year].transpose()

        oneTime = oneTime.reorder_levels(['run','model','scenario'], axis=0)
        numRecords = np.asarray(oneTime[year]).size
        scenarioColumn = list(oneTime.index.levels[2][oneTime.index.labels[2]])
        GCMcolumn = list(oneTime.index.levels[1][oneTime.index.labels[1]])
        runsColumn = list(oneTime.index.levels[0][oneTime.index.labels[0]])
        regionColumn = [region for i in xrange(numRecords)]
        periodColumn = [year for i in xrange(numRecords)]
        newOneTime = pd.DataFrame({'y': np.asarray(oneTime[year]), 'scenario': scenarioColumn, 'GCM': GCMcolumn, \
                                   'run': runsColumn, 'region': regionColumn, 'period': periodColumn})
        newOneTime.index = [i+1 for i in xrange(numRecords)]
        r_dataf = pandas2ri.py2ri(newOneTime)

        fileToSaveOut =''.join(['dataframes_R/allRCPs_',field,'_',region,'_',year,'.Rda'])
        saveRDS(r_dataf,file=fileToSaveOut)

if __name__ == "__main__":
    fields = ['tas','pr']
    regions = ['global','Alaska','BC','pnw','Cali','Baja'] 
    seasons = ['annual','DJF','JJA']
    ensembles = ['','-all+LE','-1run+LE']
    options = defaults(startYear='1950')

    for ensemble in ensembles:
        options.ensemble = ensemble
        attr.validate(options)
        for field in fields:
            for justRegion in regions:
                for season in seasons:
                    options.season=season
                    regionTag = '-'.join([justRegion,season]) 
                    regionTag = ''.join([regionTag,ensemble])
                    smoothed = getSmoothed(field,justRegion,options)
                    makeRdataframe(field,regionTag,smoothed)
