"""
SSOsoft is a set of tools for creating science-ready data from
observations with the various instruments at the Sunspot Solar
Observatory. Future releases of this package will include tools
for reducing data from all of SSO's instruments, but in this
implementation, only the ROSA and Zyla imaging instruments are
provided for.

-------------------------------------------------------------------------

Classes
-------

kisipWrapper :
    A wrapper class used for configuring and running the
    Kiepenheuer-Institut Speckle Interfrerometry Package (KISIP).
rosaZylaCal :
    A class containing all methods and attributes necessary for
    flat-fielding and formatting of images for speckle analysis
    by KISIP.
ssosoftConfig :
    Metadata showing basic information about this release of SSOsoft,
    including authorship, version, etc.
"""

import imagers

from imagers.kisipWrapper import *
from imagers.rosaZylaCal import *
from imagers.destretch import *

from spectral.hsgCal import *
