library(arm)       # Load the arm package (also loads lme4)

# Set current working directory
setwd("~/my documents/school (UW)/regional/scripts/regionalVariance/NorthropChandler2014")

# input functions MCMC.half.Cauchy.finite(), two.way.sim() and REML.results()
source("examplecode.fns")

for (which.field in c('tas','pr')) {
  for (which.regOnly in c('global','pnw','Alaska','BC','Cali','Baja')) {
    for (which.season in c('annual','DJF','JJA')) {
      for (which.runs in c('-all+LE','-1run+LE','')) {
        which.reg <- paste(which.regOnly,which.season,sep="-")
        which.reg <- paste(which.reg,which.runs,sep="")

        cat(which.field," ",which.reg,"\n")

        #=================== Climate Variable analyses (global or regional) ======================#
        for (year in 2005:2095) {
          which.period <- year          # *not* 1 for 2020-2049, 2 for 2069-2098

          fileIn = paste("../dataframes_R/allRCPs",which.field,which.reg,which.period,sep="_")
          fileIn = paste(fileIn,"Rda",sep=".")
          temp <- readRDS(file=fileIn)  # load the data
          #summary(temp)       # see which variables are present
          #temp[1:10,]         # look at first 10 rows of data

          #----------------------------------------- REML --------------------------------------#

          # fit model using REML
          fit.REML <- lmer(y~1+ (1|scenario)+(1|GCM)+(1|factor(scenario):factor(GCM)), data=temp, REML=T)

          # no confidence intervals (CI=F)
          REML.res <- REML.results(fit.REML,x1=temp$GCM,x2=temp$scenario,CI=F)
          fileOut <- paste("../NCresults/NCresults",which.field,which.reg,which.period,sep="_")

          # with confidence intervals - doing all the combos in these loops will take a long time
          #REML.res <- REML.results(fit.REML,x1=temp$GCM,x2=temp$scenario,CI=T,conf=95,n.sim=1000,my.seed=1)
          #fileOut <- paste("../NCresults/NCresults",which.field,which.reg,which.period,'Conf',sep="_")

          output <- REML.res
          saveRDS(output, file=fileOut)
        }
      }
    }
  }
}
