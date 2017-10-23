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

regionBounds = {'NewEngland':GeographicBounds(285,295,40,50),
            'caribbean':GeographicBounds(270,290,15,25), \
            'Ecuador':GeographicBounds(278,285,-5,3), \
            'amazon':GeographicBounds(300,320,-20,-10), \
            'centralAsia':GeographicBounds(70,130,45,65), \
            'Australia':GeographicBounds(120,150,-32,-20), \
            'eqAfrica':GeographicBounds(15,35,-10,15), \
            'Europe':GeographicBounds(-5,38,40,68), \
            'inland':GeographicBounds(249,279,35,49), \
            'inlandSmall':GeographicBounds(256,269,35,43), \
            'Wcoast':GeographicBounds(236,249,32,49), \
           'Alaska':GeographicBounds(193,219,58,71), \
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

years = np.arange(2011,2096)
years = [str(years[i]) for i in xrange(len(years))]
