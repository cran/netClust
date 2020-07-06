\name{netClust-package}
\alias{netClust-package}
\alias{netClust}
\docType{package}
\title{
\packageTitle{netClust}
}
\description{
\packageDescription{netClust}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{netClust}
\packageIndices{netClust}
Clustering unilayer and multilayer network data by means of finite mixtures is the main utility of 'netClust'.
}
\author{
\packageAuthor{netClust}

Maintainer: \packageMaintainer{netClust}
}
\references{
Sarkar, S. (2019) On the use of transformations for modeling multidimensional heterogeneous data, The University of Alabama Libraries Digital Collections
}

\keyword{ package }

\examples{
data(netData) ## Read network data 
data(netDataID) ## Read original ID for network data

n <- dim(netData)[1] ## number of nodes of the network
p <- dim(netData)[4] ## number of layers of the network
K <- 2               ## number of clusters 
y <- netData

eps=0.0001
RndStrtUni= 3
RndStrtMult= 5
SmEMUni= 2
SmEMMult= 3
ItrSmEM=5
burn = 10*n
ItrMCMC= 50*n
sSigma = 1
sPsi = 1
a=0

##########################################
### Run unilayer network EM on layer 1 ###
##########################################

x <- array(0, dim = c(n,n,2))
for (i in 1:n){
  for (j in 1:n){
    x[i,j,] <- y[i,j,,1]
  }
}
  
E <- netEM_unilayer(x, K, eps, RndStrtUni, SmEMUni, ItrSmEM, burn, ItrMCMC, sSigma,a)
cat("Unilayer network", "Original ID", netDataID, "\n")
cat("Unilayer network", "Assigned ID", E$id, "\n")

##################################
### Run multilayer network EM  ###
##################################

E <- netEM_multilayer(y,K,p, eps, RndStrtMult, SmEMMult, ItrSmEM, burn, ItrMCMC, sSigma, sPsi, n, a)
cat("Multilayer network", "Original ID", netDataID, "\n")
cat("Multilayer network", "Assigned ID", E$id, "\n")

}
