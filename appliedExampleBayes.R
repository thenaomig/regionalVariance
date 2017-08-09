library(arm)       # Load the arm package (also loads lme4)
library(R2WinBUGS) # Load the R2WinBUGS package

setwd("~/my documents/school (UW)/regional/scripts/regionalVariance/NorthropChandler2014")
my.bugs <- "/Users/naomi/.wine/drive_c/Program Files/WinBUGS14"
my.A <- 0.5 #see Northrup and Chandler's discussion of appropriate values

# input functions MCMC.half.Cauchy.finite(), two.way.sim() and REML.results()
source("examplecode.fns")

#for (which.field in c('tas','pr')) {
which.field <- 'pr'
for (which.regOnly in c('global','pnw','Alaska','BC','Cali','Baja')) {
  #which.regOnly <- 'pnw' #'global','pnw'
  #for (which.season in c('annual','DJF','JJA')) {
  which.season <- 'annual'    #'annual',
  #for (which.runs in c('','-all+LE','-1run+LE')) {
  which.runs <- '-all+LE'

  which.reg <- paste(which.regOnly,which.season,sep="-")
  which.reg <- paste(which.reg,which.runs,sep="")

  cat(which.field," ",which.reg,"\n")

  #=================== Climate Variable analyses (global or regional) ======================#
  for (year in 2005:2095) {
    which.period <- year          # 1 for 2020-2049, 2 for 2069-2098

    fileIn = paste("../dataframes_R/allRCPs",which.field,which.reg,which.period,sep="_")
    fileIn = paste(fileIn,"Rda",sep=".")
    temp <- readRDS(file=fileIn)  # load the data
    #summary(temp)       # see which variables are present
    #temp[1:10,]         # look at first 10 rows of data

    #----------------------------------------- REML --------------------------------------#
    set.seed(200)
    my.sim.fin.prec <- MCMC.half.Cauchy.finite(temp,A=my.A,n.iter=2500,n.chains=5,n.burnin=500,
                                               n.thin=1,bugs.directory=my.bugs,WINE="/Applications/Wine.app/Contents/Resources/bin/wine",
                                               WINEPATH="/Applications/Wine.app/Contents/Resources/bin/winepath")
    Bayes.prec.res.fin <- my.sim.fin.prec$summary[-10,]
    row.names(Bayes.prec.res.fin) <- c("mu","sigma_G","sigma_S","sigma_GS","sigma_R",
                                       "s_G","s_S","s_GS","s_R")
    output <- print(round(Bayes.prec.res.fin,3)) # summary of the results

    which.regOut <- paste(which.reg,'-Bayes',sep="")
    fileOut <- paste("../NCresults/NCresults",which.field,which.regOut,which.period,sep="_")
    saveRDS(output, file=fileOut)
        }
      }
#    }
#  }
#}
