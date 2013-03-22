MCMAC
=====

Monte Carlo Merger Analysis Code (MCMAC)

This code was originally written by Will Dawson and a major focus of his 2012 paper:
The Dynamics of Merging Clusters: A Monte Carlo Solution Applied to the Bullet and Musket
Ball Clusters (http://adsabs.harvard.edu/abs/2012arXiv1210.0014D).

The code essentially takes observed priors on each subcluster's mass, radial velocity, and
projected separation, draws randomly from those priors and uses them in a analytic model
to get posterior PDF's for merger dynamic properties of interest (e.g. collision velocity,
time since collision).

Dependencies:
Standard
- numpy
- scipy
MCC utilities (see the MCTwo GitHub account: https://github.com/MCTwo)
- profiles
- cosmo