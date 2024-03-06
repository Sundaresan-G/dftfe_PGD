import sys
sys.path.append("/home/srinibasnandi/project_works/bandsPR/dftfe") # append the path of DFTFE to sys.path

from postProcessing.postprocessModules import Plotters
plotter = Plotters("./","bands.out", "kpointRuleFile.inp", "coordinatesRelaxed.inp", "domainVectorsRelaxed.inp","pseudo.inp","dosData.out", [-14.0, -5.0], True)
# plotter = Plotters("./","bands.out", "kpointRuleFile.inp", "coordinatesRelaxed.inp", "domainVectorsRelaxed.inp","pseudo.inp","dosData.out", [-20.0, 20.0], True)
plotter.plotBandStr()
plotter.plotDos()
