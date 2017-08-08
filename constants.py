import numpy as np

# Global variables
directoryCMIP = '/Users/naomi/my documents/school (UW)/regional/data/'
directoryLE = directoryCMIP
scenarios = ['rcp85','rcp60','rcp45','rcp26']
scenariosLE = ['rcp85'] #,'rcp45'] 

years = np.arange(2005,2096)
years = [str(years[i]) for i in xrange(len(years))]
