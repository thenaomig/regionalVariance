import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib
import string
import os.path
import statsmodels.api as sm
from copy import deepcopy
import attr
from constants import * #<= my defined global variables

import rpy2.robjects as ro
Rload = ro.r('readRDS')
from rpy2.robjects import pandas2ri
pandas2ri.activate()

#Helper functions
def allStartJanuary(allRCPs):
    for i, rcp in enumerate(allRCPs):
        while allRCPs.loc[:,rcp].dropna().index[0].month != 1:
            rowIndex = allRCPs.loc[:,rcp].dropna().index[0]
            allRCPs.loc[rowIndex,rcp] = np.NaN
    return allRCPs

def getFieldLE(field):
    if field == 'tas':
        fieldLE = 'TS'
    elif field == 'pr':
        fieldLE = 'PRECT'
    return fieldLE

def subtractTemp(y):
    return y - y.transpose()['1970':'1999'].transpose().mean().transpose().mean()

#Define default options class and various validation functions.------------------
def nRowChoices(instance,attribute,value):
    if value == 2:
        instance.varianceRow = 0
    elif value == 3:
        instance.varianceRow = 1
    else:
        raise ValueError("Plotting functions only support 2 or 3 row options.")

def consistentFlags(instance,attribute,value):
    if value == '':
        if instance.HandS == False:
            instance.oneCMIPrun = False
            instance.justCMIP = True
    elif value == '-all+LE':
        instance.oneCMIPrun = False
        instance.justCMIP = False
    elif value == '-1run+LE':#
        instance.oneCMIPrun = True
        instance.justCMIP = False
    else:
        raise ValueError("'ensemble' must be one of ['','-all+LE','-1run+LE']")

def seasonChoices(instance,attribute,value):
    #could be extended to add more, but need to put options in 'readInCMIPandLE'
    if value not in ['annual','DJF','JJA']:
        raise ValueError("'season' must be one of ['annual','DJF','JJA']")

def estimateLowerUpper(instance,attribute,value):
    if value not in ['estimate','lower','upper']:
        raise ValueError("'which' must be one of ['estimate','lower','upper']")
    if instance.confidenceIntervals == False:
        instance.which = 'estimate'

def isRCP(instance,attribute,value):
    if not isinstance(value,list):
        raise ValueError("'scenarios' must be instance of type list")
    for item in value:
        if item not in ['rcp85','rcp26','rcp60','rcp45']:
            raise ValueError("List items are limited to ['rcp85','rcp26','rcp60','rcp45']")

def consistentHandS(instance,attribute,value):
    if value == True:
        instance.ensemble = ''
        instance.startYear = '1950'
        instance.oneCMIPrun = True
        instance.justCMIP = True
        instance.confidenceIntervals = False

def isHandSoption(instance,attribute,value):
    if value not in ['smoothed','default','thirdOrder','quadratic','linear']:
        raise ValueError("'internal' must be one of ['smoothed','default','thirdOrder', \
        'quadratic','linear'] to calculate internal variance in 'getVariancesHandS'")

def listOfTwo(instance,attribute,value):
    #it's okay if this is None, but if it's something, it better be a list of two
    if value:
        if not isinstance(value,list):
            raise ValueError("limits must be instance of type list")
        if len(value) != 2:
            raise ValueError("axis limits must be list of two elements [min,max]")

@attr.s
class defaults(object):
    nRows = attr.ib(default=2,validator=nRowChoices)
    varianceRow = attr.ib(default=0)
    season = attr.ib(default='annual',convert=str,validator=seasonChoices)
    landOnly = attr.ib(default='')
    ensemble = attr.ib(attr.Factory(str),validator=consistentFlags)
    append = attr.ib(default='',convert=str)
    startYear = attr.ib(default='2006',convert=str)
    oneCMIPrun = attr.ib(attr.Factory(bool))
    justCMIP = attr.ib(attr.Factory(bool))
    confidenceIntervals = attr.ib(default=False)
    which = attr.ib(default='estimate',convert=str,validator=estimateLowerUpper)
    smooth = attr.ib(default=None)
    ylimVariance = attr.ib(default=None,validator=listOfTwo)
    ylimMeanRow = attr.ib(default=None,validator=listOfTwo)
    tick_label_size = attr.ib(default='small')
    spaghetti = attr.ib(default=False)
    scenarios = attr.ib(default=['rcp85','rcp26'],validator=isRCP)
    HandS = attr.ib(default=False,validator=consistentHandS)
    scenariosToDrop = attr.ib(default=[],validator=isRCP)
    modelsToKeep = attr.ib(default=[],validator=attr.validators.instance_of(list))
    internal = attr.ib(default='smoothed',validator=isHandSoption)

#----------------------------------------------------------------
def readInAllTimeSeries(whichData,directoryData,scenarios,field,options):
    numScenarios = len(scenarios)

    if('cesmLE' in whichData):
        oneRCPhist = pd.DataFrame()
    else:
        fileName = ''.join(['timeSeries',options.landOnly,'_',field,'_',whichData,'_historical.csv'])
        fName = ''.join([directoryData,fileName])
        oneRCPhist = pd.read_csv(fName, index_col=0, header=[0,1], parse_dates=True)
        #get the names of the models from the first scenario file in the list:
        models = np.asarray(oneRCPhist.columns) #<- all that just to get the model list, and historical data

    def getOneFrame(i):
        if('cesmLE' in whichData):
            fileName = ''.join(['timeSeries',options.landOnly,'_',field,'_',whichData,'_',scenarios[i],'.csv'])
            fName = ''.join([directoryData,fileName])
            oneRCP = pd.read_csv(fName, index_col=0, parse_dates=True)
            oneRCP = oneRCP.transpose()
            return oneRCP
        else:
            fileName = ''.join(['timeSeries',options.landOnly,'_',field,'_',whichData,'_',scenarios[i],'.csv'])
            fName = ''.join([directoryData,fileName])
            oneRCP = pd.concat([oneRCPhist, pd.read_csv(fName, index_col=0, header=[0,1], parse_dates=True)])
            oneRCP = oneRCP.transpose()
            return oneRCP

    allRCPs = pd.concat([ getOneFrame(i) for i in xrange(numScenarios)],axis=0,keys=scenarios)

    #build the multiIndex for the dataframe to hold it all:
    if ('ensemble' in whichData or 'cesmLE' in whichData):
        indexToUse = pd.MultiIndex.from_tuples(allRCPs.transpose(),names=['scenario','model'])
    else:
        indexToUse = pd.MultiIndex.from_tuples(allRCPs.transpose(),names=['scenario','model','run'])
    allRCPs.index = indexToUse #do assign the constructed multi-index, need these as the row titles
    allRCPs = allRCPs.transpose() #make the time index the rows again

    if ('cesmLE' not in whichData):
        allRCPs = allStartJanuary(allRCPs)
        #clean dataframe of bad data - could just be my versions that are corrupted
        allRCPs = allRCPs.drop(('rcp85','EC-EARTH','run7'),axis=1)
        allRCPs = allRCPs.drop(('rcp85','EC-EARTH','run8'),axis=1)
        allRCPs = allRCPs.drop(('rcp85','EC-EARTH','run9'),axis=1)
        allRCPs = allRCPs.drop(('rcp85','EC-EARTH','run11'),axis=1)
        allRCPs = allRCPs.drop(('rcp60','EC-EARTH','run7'),axis=1)
        allRCPs = allRCPs.drop(('rcp45','EC-EARTH','run7'),axis=1)
        allRCPs = allRCPs.drop(('rcp26','EC-EARTH','run7'),axis=1)
        allRCPs = allRCPs.drop(('rcp85','CESM1-CAM5','run3'),axis=1)
        allRCPs = allRCPs.drop(('rcp45','EC-EARTH','run5'),axis=1)
        if field == 'pr':
            allRCPs = allRCPs.drop(('rcp45','CSIRO-Mk3L-1-2','run1'),axis=1)
        #and all nans:
        for num in np.arange(2,10):
            allRCPs = allRCPs.drop(('rcp45','MIROC-ESM-CHEM','run'+str(num)),axis=1)

    return allRCPs

def readInCMIPandLE(field,region,options):
    fieldLE = getFieldLE(field)
    whichData = region+'_Monthly'
    whichDataLE = whichData+'_cesmLE'

    allRCPs = readInAllTimeSeries(whichData,directoryCMIP,scenarios,field,options)
    cesmLE = readInAllTimeSeries(whichDataLE,directoryLE,scenariosLE,fieldLE,options)
    cesmLE.columns = cesmLE.columns.set_names(['scenario', 'run'])

    if options.season == 'annual':
        cesmLE = cesmLE.resample('1A',closed='right',label='left',loffset='1D').mean()
        allRCPs = allRCPs.resample('1A',closed='right',label='left',loffset='1D').mean()
    elif options.season == 'DJF':
        cesmLE = cesmLE.resample('3M',loffset='1M').mean()
        cesmLE = cesmLE[1:-1]
        cesmLE = cesmLE.loc[cesmLE.index.month==2] #month is last month of season
        allRCPs = allRCPs.resample('3M',loffset='1M').mean()
        allRCPs = allRCPs[1:-1]
        allRCPs = allRCPs.loc[allRCPs.index.month==2] #month is last month of season
    elif options.season == 'JJA':
        cesmLE = cesmLE.resample('3M',loffset='1M').mean()
        cesmLE = cesmLE[1:-1]
        cesmLE = cesmLE.loc[cesmLE.index.month==8]
        allRCPs = allRCPs.resample('3M',loffset='1M').mean()
        allRCPs = allRCPs[1:-1]
        allRCPs = allRCPs.loc[allRCPs.index.month==8]

    allRCPs.index = allRCPs.index.to_period(freq='A')
    cesmLE.index = cesmLE.index.to_period(freq='A')

    if options.oneCMIPrun:
        allRCPs = allRCPs.swaplevel(0,2,axis=1)
        allRCPs = allRCPs['run1'] #drop everything other than one of the runs
        allRCPs = allRCPs.swaplevel(0,1,axis=1)
        if options.justCMIP:
            return allRCPs
        #but keep the 'run' level in the column headers:
        allRCPs = allRCPs.transpose()
        allRCPs['1772'] = 'run1'
        allRCPs.set_index('1772',append=True,inplace=True)
        allRCPs = allRCPs.transpose()
        allRCPs.columns = allRCPs.columns.set_names(['scenario','model','run'])

    if options.justCMIP:
        return allRCPs

    cesmLE = cesmLE.transpose()
    cesmLE['1492'] = 'CESM-LE' #<- with placeholder name as a hack so I can add a column
    cesmLE.set_index('1492', append=True, inplace=True)
    cesmLE.reorder_levels(['scenario', '1492', 'run'])
    cesmLE = cesmLE.transpose()
    cesmLE.columns = cesmLE.columns.set_names(['scenario', 'run','model'])

    cesmLE = cesmLE.swaplevel(1,2,axis=1)
    toDrop = ['106','107','OIC001d','OIC002','OIC003','OIC004','OIC005','OIC006','OIC007','OIC008','OIC009','OIC010']
    for i, model in enumerate(toDrop):
        cesmLE = cesmLE.transpose().drop(toDrop[i],level='run').transpose()
    cesmLE = cesmLE.swaplevel(0,1,axis=1)

    allRCPsPlus = deepcopy(cesmLE)
    allRCPs = allRCPs.swaplevel(0,1,axis=1)
    allRCPsPlus = pd.concat([allRCPsPlus,allRCPs],axis=1)

    allRCPsPlus = allRCPsPlus.swaplevel(0,1,axis=1)

    return allRCPsPlus

def getAnomaly(field,region,options):
    allRCPsPlus = readInCMIPandLE(field,region,options)
    anom = deepcopy(allRCPsPlus)
    
    for scenario in scenarios: #subtract the right historical average, across all runs of that model/scenario
        anom[scenario] = allRCPsPlus[scenario].transpose().groupby('model').apply(subtractTemp).transpose()

    toNotDrop = anom['2000':'2099'].transpose().dropna().transpose().columns.values.tolist()
    anom = anom.loc[options.startYear:'2099',toNotDrop]
    return anom

def getSmoothed(field,region,options):
    anom = getAnomaly(field,region,options)
    #uses a ten year running mean
    smoothed = anom.rolling(min_periods=10,window=10,center=True).mean()
    return smoothed

def readNCresults(field,region,smoothed,options):
    '''Read back in the results of running the R scripts
    from Northrop and Chandler (2014).
    ['mu','sigma_G','sigma_S','sigma_GS','sigma_R']'''
    sigmaG = np.zeros(len(years))
    sigmaS = np.zeros(len(years))
    sigmaGS = np.zeros(len(years))
    sigmaR = np.zeros(len(years))
    for i, year in enumerate(years):
        if options.confidenceIntervals: #with confidence intervals
            fileBackIn = '_'.join(['NCresults/NCresults',field,region,year,'Conf'])
        else:
            fileBackIn = '_'.join(['NCresults/NCresults',field,region,year])
        try:
            NCresults = Rload(fileBackIn)
        except:
            raise IOError("couldn't find ", fileBackIn)
        NCresultsPy = pandas2ri.ri2py(NCresults)
        if options.confidenceIntervals or ('Bayes' in region):
            if options.which=='estimate':
                element = 0
            elif options.which == 'lower':
                element = 2
            elif options.which == 'upper':
                element = 3
            sigmaG[i] = np.power(NCresultsPy[1][element],2)
            sigmaS[i] = np.power(NCresultsPy[2][element],2)
            sigmaGS[i] = np.power(NCresultsPy[3][element],2)
            sigmaR[i] = np.power(NCresultsPy[4][element],2)
        else:
            sigmaG[i] = np.power(NCresultsPy[1],2)
            sigmaS[i] = np.power(NCresultsPy[2],2)
            sigmaGS[i] = np.power(NCresultsPy[3],2)
            sigmaR[i] = np.power(NCresultsPy[4],2)

    if options.smooth:
        fracForSmoothing = options.smooth
        leadTime = pd.DataFrame(np.arange(0,len(years)),index=smoothed['2011':'2095'].index)
        leadTime.columns = ['time']
        sigmaG = sm.nonparametric.lowess(sigmaG,np.array(leadTime.transpose()).astype('timedelta64[Y]')[0],frac=fracForSmoothing,it=0,return_sorted=False)
        sigmaS = sm.nonparametric.lowess(sigmaS,np.array(leadTime.transpose()).astype('timedelta64[Y]')[0],frac=fracForSmoothing,it=0,return_sorted=False)
        sigmaGS = sm.nonparametric.lowess(sigmaGS,np.array(leadTime.transpose()).astype('timedelta64[Y]')[0],frac=fracForSmoothing,it=0,return_sorted=False)
        sigmaR = sm.nonparametric.lowess(sigmaR,np.array(leadTime.transpose()).astype('timedelta64[Y]')[0],frac=fracForSmoothing,it=0,return_sorted=False)

    modelComponent = pd.Series(sigmaG,index=smoothed['2011':'2095'].index)
    scenarioComponent = pd.Series(sigmaS,index=smoothed['2011':'2095'].index)
    interactionComponent = pd.Series(sigmaGS,index=smoothed['2011':'2095'].index)
    internalComponent = pd.Series(sigmaR,index=smoothed['2011':'2095'].index)

    return modelComponent, scenarioComponent, interactionComponent, internalComponent

def getVariancesHandS(region,rcpsIn,options,**kwargs):
    '''This is the function that emulates the method of Hawkins and Sutton (2009).
    Returns three components of uncertainty as a function of time, and the highly
    smoothed time series that went into the calculation.
    Parameters specific to this function that can be specified in options include
    'modelsToKeep','scenariosToDrop',and 'internal' to specify the smoothing method.'''
    whichData = region+'_Monthly'
    allRCPs1950on = deepcopy(rcpsIn)
    allRCPs1950on = allRCPs1950on.transpose().dropna().transpose()

    oneModelName = 'GFDL-ESM2G'
    if bool(options.scenariosToDrop):
        for toDrop in options.scenariosToDrop:
            allRCPs1950on = allRCPs1950on.drop(toDrop,axis=1,level='scenario')
    #get the list of remaining scenarios from one model that we know had all 4 to begin with
    scenarios = allRCPs1950on.swaplevel('scenario','model', axis=1)[oneModelName].columns.values
    numScenarios = len(scenarios)
    #===============================================================

    #only use the models which exist for all scenarios==============
    #group by model
    byModel = allRCPs1950on['20000101'::].transpose().groupby(level='model')
    #construct a model list
    modelList = list(allRCPs1950on.columns.levels[1])
    numModels = len(modelList)
    #if a modelsToKeep was specified, first get rid of the models not in that list
    if (len(np.array(options.modelsToKeep)) != 0):
        for model in modelList:
            if model not in options.modelsToKeep:
                modelList.remove(model)
                #smoothed = smoothed.drop(model,axis=1,level='model')
                allRCPs1950on = allRCPs1950on.drop(model,axis=1,level='model')
    #then make sure we only use models that exist for all scenarios
    modelsToRemove = []
    for model in modelList:
        scenariosThisModel = allRCPs1950on.swaplevel(0,1,axis=1)[model].columns.values
        for scenario in scenarios:
            if (scenario not in scenariosThisModel) and (model not in modelsToRemove):
                modelsToRemove.append(model)
    for model in modelsToRemove:
        modelList.remove(model)
        allRCPs1950on = allRCPs1950on.drop(model,axis=1,level='model')

    numModels = len(modelList)

    #get the uncertainty due to internal variability================
    def getResiduals(scenario,model):
        oneScenario = allRCPs1950on[scenario]
        y = deepcopy(oneScenario[model])
        y = y.dropna()

        X = pd.DataFrame((np.arange(0,len(y.index))))
        leadTime = np.arange(0,len(y.index))
        X.index = y.index
        X = X.transpose()
        if(options.internal=='default' or options.internal=='smoothed'):
            X = pd.concat([X,pow(X,2),pow(X,3),pow(X,4)],keys=['x1','x2','x3','x4'])
        elif(options.internal=='thirdOrder'):
            X = pd.concat([X,pow(X,2),pow(X,3)],keys=['x1','x2','x3'])
        elif(options.internal=='quadratic'):
            X = pd.concat([X,pow(X,2)],keys=['x1','x2'])
        elif(options.internal=='linear'):
            X = pd.concat([X],keys=['x1'])
        X = X.transpose()
        X = sm.add_constant(X)
        results = sm.OLS(y,X).fit()
        results.summary()
        #alt thing:
        if options.internal=='smoothed':
            lowessSmoothedMore = sm.nonparametric.lowess(np.asarray(y),leadTime,frac=0.4,it=0,return_sorted=False)
            allRCPs1950on.loc[y.index,(scenario,model)] = lowessSmoothedMore
            residual = np.asarray(y - lowessSmoothedMore)
        else:
            allRCPs1950on.loc[y.index,(scenario,model)] = results.fittedvalues
            residual = np.asarray(y - results.fittedvalues)

        #yes, a decadal smoothing of the residuals themselves. See H&S '09 supplement, p. 1104
        lowessSmoothed = sm.nonparametric.lowess(residual,leadTime,frac=0.06667,it=0,return_sorted=False)
        return np.asarray(lowessSmoothed)

    def getVarOneModel(m):
        residuals = np.concatenate([ getResiduals(scenarios[i],modelList[m]) for i in xrange(numScenarios)])
        return np.power(residuals.std(),2) #one number for variance across scenario and time

    allVarOfResids = []
    leadTime = np.arange(0,len(allRCPs1950on.index))
    for m in xrange(numModels):
        allVarOfResids.append(getVarOneModel(m))
    print 'internal variance H&S',options.internal,'method, N=',len(allRCPs1950on.columns),': ', np.mean(allVarOfResids)

    #get the uncertainty due to intermodel variability =============
    byScenario = allRCPs1950on['20000101'::].transpose().groupby(level='scenario')
    modelComponent = np.mean(np.power(byScenario.std(),2))
    #=======and the scenario chosen=================================
    scenarioComponent = np.power(np.std(byScenario.mean()),2)
    #===============================================================

    internalComponent = deepcopy(modelComponent)
    internalComponent[:] = np.mean(allVarOfResids) #assumes this stays constant over time
    #===============================================================
    return internalComponent, modelComponent, scenarioComponent, allRCPs1950on['20000101'::]

#---------------------------------------------------------
#---Functions related to plotting-------------------------
def plotTrends(ax,column,smoothed,options,**kwargs):
    plt.axes(ax[0,column])

    smoothed = smoothed['20110101':'20950101']

    byScenario = smoothed.transpose().groupby(level='scenario')
    mean_field = byScenario.mean().transpose()

    colorList = ['tomato','lightblue','g','purple']
    plt.rc('axes', color_cycle=colorList)
    if not options.spaghetti:
        mstd = byScenario.std().transpose()
        for i, scenario in enumerate(options.scenarios):
            mean_field[scenario].plot(color=colorList[i],linewidth=2)
            plt.fill_between(mean_field.index,mean_field[scenario]-mstd[scenario],mean_field[scenario]+mstd[scenario],facecolor=colorList[i],alpha=0.2)
    else:
        for i,scenario in enumerate(options.scenarios):
            for j in smoothed[scenario].columns:
                smoothed[scenario][j].plot(color=colorList[i],linewidth=0.5)

    forLegend = byScenario.groups.keys()
    plt.tick_params(labelsize=options.tick_label_size)
    plt.tick_params(axis='x',label1On=False)
    plt.xlim(['2006','2099'])
    plt.xlabel('')
    if(column==0):
        plt.ylabel('Multi-model means')
    else:
        plt.tick_params(labelleft='off')

def plotVariance(ax,column,options,**kwargs):
    smoothed = kwargs['smoothed']
    smoothed = smoothed['20110101':'20950101']

    modelComponent = kwargs['modelComponent']['20110101':'20950101']
    scenarioComponent = kwargs['scenarioComponent']['20110101':'20950101']
    internalComponent = kwargs['internalComponent']['20110101':'20950101']

    #total variance
    plt.axes(ax[options.varianceRow,column])
    modelComponent.plot(color='b',linewidth=2)
    scenarioComponent.plot(color='g',linewidth=2)
    internalComponent.plot(color='orange',linewidth=2)

    if 'interactionComponent' in kwargs:
        interactionComponent = kwargs['interactionComponent']['20110101':'20950101']
        totalVariance = internalComponent + modelComponent + scenarioComponent + interactionComponent
        interactionComponent.plot(color='c', linewidth=2)
    else:
        totalVariance = internalComponent + modelComponent + scenarioComponent

    totalVariance.plot(color='k',linewidth=2)
    plt.xlim(['2006', '2099'])
    plt.tick_params(labelsize=options.tick_label_size)
    plt.xticks(['2010','2030','2050','2070','2090'])
    plt.tick_params(labelbottom='off')
    plt.xlabel('')
    if(column == 0):
        plt.ylabel('Variance')
    else:
        plt.tick_params(labelleft='off')

    fracModel = (modelComponent/totalVariance)
    fracScenario = (scenarioComponent/totalVariance)
    fracInternal = (internalComponent/totalVariance)

    if 'interactionComponent' in kwargs:
        fracInteraction = (interactionComponent/totalVariance)
    else:
        fracInteraction = deepcopy(fracInternal)
        fracInteraction[:] = 0.0

    zeroLine = deepcopy(fracModel)
    zeroLine[:] = 0.0
    onesLine = deepcopy(zeroLine)
    onesLine[:] = 1.0

    #fractional variance: H&S order
    plt.axes(ax[options.varianceRow+1,column])
    xDim = pd.Series(np.arange(2011,2096)) 
    plt.fill_between(xDim,fracModel+fracScenario+fracInteraction,onesLine,facecolor='orange')
    plt.fill_between(xDim,fracModel+fracInteraction,fracModel+fracInteraction+fracScenario,facecolor='green')
    plt.fill_between(xDim,fracModel,fracModel+fracInteraction,facecolor='cyan')
    plt.fill_between(xDim,zeroLine,fracModel,facecolor='blue')

    plt.xlim([2006, 2099])
    if(column == 0):
        plt.ylabel('Fraction of Variance')
    else:
        plt.tick_params(labelleft='off')
    plt.yticks(np.arange(10)/10.0 + 0.1)
    plt.tick_params(labelsize=options.tick_label_size)
    plt.tick_params(axis='x',direction='inout')
    plt.xticks([2010,2030,2050,2070,2090],rotation=25)

def plotVarianceWithIntervals(ax,column,options,**kwargs):
    '''These are optionally plotted on top of already-plotted best estimates, if
    this function is called.
    This interval-plotting add-on will always have an interaction component specified
    because the confidence intervals are only produced by the method that also
    has an interaction term.'''
    smoothed = kwargs['smoothed']

    internalLow = kwargs['internalLow']['20110101':'20950101']
    modelLow = kwargs['modelLow']['20110101':'20950101']
    scenarioLow = kwargs['scenarioLow']['20110101':'20950101']
    interactionLow = kwargs['interactionLow']['20110101':'20950101']

    internalHigh = kwargs['internalHigh']['20110101':'20950101']
    modelHigh = kwargs['modelHigh']['20110101':'20950101']
    scenarioHigh = kwargs['scenarioHigh']['20110101':'20950101']
    interactionHigh = kwargs['interactionHigh']['20110101':'20950101']

    plt.axes(ax[options.varianceRow,column])
    plt.tick_params(labelsize=options.tick_label_size)

    modelLow.plot(color='b')
    modelHigh.plot(color='b')
    scenarioLow.plot(color='g')
    scenarioHigh.plot(color='g')
    internalLow.plot(color='orange')
    internalHigh.plot(color='orange')
    interactionLow.plot(color='c')
    interactionHigh.plot(color='c')

    plt.xlim(['2006', '2099'])
    plt.xlabel('')

def plotColumn(ax,column,field,region,options):
    if options.HandS:
        anomaly = getAnomaly(field,region,options)
        internalComponent, modelComponent, scenarioComponent, smoothed = getVariancesHandS(region,anomaly,options)
        components = {'modelComponent':modelComponent,'scenarioComponent':scenarioComponent,
                         'internalComponent':internalComponent,'smoothed':smoothed}
    else:
        smoothed = getSmoothed(field,region,options)
        regionTag = '-'.join([region,options.season])
        regionTag = ''.join([regionTag,options.ensemble,options.append])
        if options.landOnly=='LO':
            regionTag = '-'.join([regionTag,options.landOnly])
        modelComponent, scenarioComponent, interactionComponent, internalComponent = readNCresults(field,regionTag,smoothed,options)
        components = {'modelComponent':modelComponent,'scenarioComponent':scenarioComponent,
                     'interactionComponent':interactionComponent,'internalComponent':internalComponent,
                     'smoothed':smoothed}
    if options.nRows == 3:
        plotTrends(ax,column,smoothed,options)
        if options.ylimMeanRow:
            plt.axes(ax[0,column])
            plt.ylim(options.ylimMeanRow)

    plotVariance(ax,column,options,**components)

    if options.confidenceIntervals:
        options.which='lower'
        modelLow, scenarioLow, interactionLow, internalLow = readNCresults(field,regionTag,smoothed,options)
        options.which='upper'
        modelHigh, scenarioHigh, interactionHigh, internalHigh = readNCresults(field,regionTag,smoothed,options)
        componentsMore = {'modelLow':modelLow,'scenarioLow':scenarioLow,
                 'interactionLow':interactionLow,'internalLow':internalLow,
                 'modelHigh':modelHigh,'scenarioHigh':scenarioHigh,
                 'interactionHigh':interactionHigh,'internalHigh':internalHigh}
        components.update(componentsMore)
        plotVarianceWithIntervals(ax,column,options,**components)
        options.which='estimate' #<=reset

        print "internal variance CMIP5",options.ensemble,", N=",len(smoothed.columns),": ", internalComponent.mean(), " range: ", internalLow.mean(), internalHigh.mean()
    #else:
    #    print "internal variance CMIP5",options.ensemble,", N=",len(smoothed.columns),": ", internalComponent.mean()

    if options.ylimVariance:
        plt.axes(ax[options.varianceRow,column])
        plt.ylim(options.ylimVariance)

def letterSubfigures(ax):
    for n, oneax in enumerate(ax.flat):
        oneax.text(-0.05, 1.05, string.ascii_lowercase[n], transform=oneax.transAxes,
                size=14, weight='bold')

def plotLegend(fig):
    tomato_line = mlines.Line2D([],[],color='tomato',linewidth=2,linestyle='--')
    purple_line = mlines.Line2D([],[],color='purple',linewidth=2,linestyle='--')
    green_line = mlines.Line2D([],[],color='g',linewidth=2,linestyle='--')
    blue_line = mlines.Line2D([],[],color='lightblue',linewidth=2,linestyle='--')
    fig.legend(handles=[tomato_line,purple_line,green_line,blue_line],labels=['RCP 8.5','RCP 6.0','RCP 4.5','RCP 2.6'],loc='upper right')

    blue_patch = mpatches.Patch(color='b')
    cyan_patch = mpatches.Patch(color='c')
    green_patch = mpatches.Patch(color='g')
    orange_patch = mpatches.Patch(color='orange')
    figLabels = ['Internal','Scenario','Interaction','GCM']
    fig.legend(handles=[orange_patch,green_patch,cyan_patch,blue_patch],labels=figLabels,loc='lower right')
