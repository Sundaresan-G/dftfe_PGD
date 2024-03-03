import sys
sys.path.append("/home/srinibasnandi/project_works/bandsPR/dftfe") # append the path of DFTFE to sys.path

from postProcessing.plotBandstructure import plotBandStr

plotBandStr("./","bands.out", "kpointRuleFile.inp", "coordinatesRelaxed.inp", "domainVectorsRelaxed.inp", [-14.0, -5.0], True)