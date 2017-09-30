#Effectively, this is where all global variables are declared.

import numpy as np

# Global variables for Step 1.
dataDirCESM = '/directory/path/to/monthly/data/'
dataDirCMIP = '/directory/path/for/CMIP/'
weightPathAndPrefix = '/directory/path/to/CMIP5/areacella/areacella_fx_'
weightSuffix = '_historical_r0i0p0.nc'
landFracPath = '/directory/path/to/landFrac/sftlf/'
weightFileCESM = '/any/path/to/file/like/rcp85/monthly/TS/b.e11.BRCP85C5CNBDRD.f09_g16.008.cam.h0.TS.208101-210012.nc'
landFracCESM = '/directory/path/to/CESM/LANDFRAC/'

class GeographicBounds(object):
    def __init__(self, lonMin, lonMax, latMin, latMax):
        self.lonMin = lonMin
        self.lonMax = lonMax
        self.latMin = latMin
        self.latMax = latMax

regionBounds = {'Alaska':GeographicBounds(193,219,58,71), \
           'Cali':GeographicBounds(236,245,32.5,42), \
            'BC':GeographicBounds(226,240,49,60), \
            'Baja':GeographicBounds(243, 251, 23, 32.5), \
           'pnw':GeographicBounds(236, 249, 42, 49), \
            'global':GeographicBounds(0,360,-90,90)}

# Global variables
directoryCMIP = 'timeSeries/' #for area-averaged time series
directoryLE = directoryCMIP
scenarios = ['rcp85','rcp60','rcp45','rcp26']
scenariosLE = ['rcp85'] #,'rcp45']

years = np.arange(2005,2096)
years = [str(years[i]) for i in xrange(len(years))]
