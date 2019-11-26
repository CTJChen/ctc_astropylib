from __future__ import print_function, division
import numpy as np
from numpy import pi, sqrt, exp, meshgrid, shape, ones
from PyAstronomy.funcFit.onedfit import OneDFit
from PyAstronomy.pyaC import pyaErrors as PE

class KingFit2d(OneDFit):
  """
    Implements a two dimensional Gaussian.
    
    Expects a coordinate array to evaluate model.
    
    The functional form is:
  
    .. math:: \\frac{A}{2\\pi\\sigma_x\\sigma_y\\sqrt{1-\\rho^2}}
              exp\\left(-\\frac{1}{2(1-\\rho^2)}\left( \\frac{(x-\\mu_x)^2}{\\sigma_x^2} +
              \\frac{(y-\\mu_y)^2}{\\sigma_y^2} -
              \\frac{2\\rho(x-\\mu_x)(y-\\mu_y)}{\\sigma_x\\sigma_y}
              \\right)\\right)
    
    Here, `lin` and `off` denote the linear and the offset term.
    
    *Fit parameters*:
     - `A` - Amplitude (the area of the Gaussian)
     - `x0` - Center of the King's Function (x-axis)
     - `y0` - Center of the King's Function (y-axis)
     - `sig_core` - "size of the King function core"
     - `sig_wing` - "size of the King function wing"
     - `f_core` - core component normalization, f_wing = 1 - f_core
  """
  
  def __init__(self):
    OneDFit.__init__(self, ["A", "sig_core", "sig_wing", "x0", "y0", "f_core"])
    self.setRootName("KingFit2d")

  def evaluate(self, co):
    """
      Evaluates the model for current parameter values.
      
      Parameters
      ----------
      co : array
           Specifies the points at which to evaluate the model.
    """

    if (self["sig_core"] <= 0.0) or (self["sig_wing"] <= 0.0):
      raise(PE.PyAValError("Width(s) of Gaussian must be larger than zero.", \
                           solution="Change width ('sig_core/y')."))
    if self["f_core"] > 1.0:
      raise(PE.PyAValError("The relative normalization of the core and wing components shuold be between 0 and 1.", \
                           solution="Set limits of f_core."))
    f_wing = 1 - self["f_core"]
    result = self["A"] * ( \
      self["f_core"] / \
    ((1 + ((co[::,::,0] - self["x0"])**2 + (co[::,::,1] - self["y0"])**2)/(self["sig_core"]**2)\
      ) ** 2) + \
    f_wing / \
    ((1 + ((co[::,::,0] - self["x0"])**2 + (co[::,::,1] - self["y0"])**2)/(self["sig_wing"]**2)\
      ) ** 2)
    )
    return result


class GaussFit2dConst(OneDFit):
  """
    Implements a two dimensional Gaussian.
    
    Expects a coordinate array to evaluate model.
    
    The functional form is:
  
    .. math:: \\frac{A}{2\\pi\\sigma_x\\sigma_y\\sqrt{1-\\rho^2}}
              exp\\left(-\\frac{1}{2(1-\\rho^2)}\left( \\frac{(x-\\mu_x)^2}{\\sigma_x^2} +
              \\frac{(y-\\mu_y)^2}{\\sigma_y^2} -
              \\frac{2\\rho(x-\\mu_x)(y-\\mu_y)}{\\sigma_x\\sigma_y}
              \\right)\\right) + constant
    
    Here, `lin` and `off` denote the linear and the offset term.
    
    *Fit parameters*:
     - `A` - Amplitude (the area of the Gaussian)
     - `mux` - Center of the Gaussian (x-axis)
     - `muy` - Center of the Gaussian (y-axis)
     - `sigx` - Standard deviation (x-axis)
     - `sigy` - Standard deviation (y-axis)
     - `rho` - Correlation
     - `const` - constant
  """
  
  def __init__(self):
    OneDFit.__init__(self, ["A", "mux", "muy", "sigx", "sigy", "rho", "const"])
    self.setRootName("GaussFit2dConst")

  def evaluate(self, co):
    """
      Evaluates the model for current parameter values.
      
      Parameters
      ----------
      co : array
           Specifies the points at which to evaluate the model.
    """
    if (self["sigx"] <= 0.0) or (self["sigy"] <= 0.0):
      raise(PE.PyAValError("Width(s) of Gaussian must be larger than zero.", \
                           solution="Change width ('sigx/y')."))
    if self["rho"] > 1.0:
      raise(PE.PyAValError("The correlation coefficient must be 0 <= rho <= 1.", \
                           solution="Change width ('sigx/y')."))
    result = self["A"]/(2.*pi*self["sigx"]*self["sigy"]*sqrt(1.-self["rho"]**2)) * \
        exp( ((co[::,::,0]-self["mux"])**2/self["sigx"]**2 + (co[::,::,1]-self["muy"])**2/self["sigy"]**2 - \
            2.*self["rho"]*(co[::,::,0]-self["mux"])*(co[::,::,1]-self["muy"])/(self["sigx"]*self["sigy"])) / \
            (-2.*(1.-self["rho"]**2)) ) + self['const']
    return result

class EllipGauss2dConst(OneDFit):
  """
    Implements a two dimensional Gaussian.
    
    Expects a coordinate array to evaluate model.
    
    The functional form is:
  
    .. math:: \\frac{A}{2\\pi\\sigma_x\\sigma_y\\sqrt{1-\\rho^2}}
              exp\\left(-\\frac{1}{2(1-\\rho^2)}\left( \\frac{(x-\\mu_x)^2}{\\sigma_x^2} +
              \\frac{(y-\\mu_y)^2}{\\sigma_y^2} -
              \\frac{2\\rho(x-\\mu_x)(y-\\mu_y)}{\\sigma_x\\sigma_y}
              \\right)\\right) + constant
    
    Here, `lin` and `off` denote the linear and the offset term.
    
    *Fit parameters*:
     - `A` - Amplitude (the area of the Gaussian)
     - `mux` - Center of the Gaussian (x-axis)
     - `muy` - Center of the Gaussian (y-axis)
     - `sigx` - Standard deviation (x-axis)
     - `sigy` - Standard deviation (y-axis)
     - `rho` - Correlation
     - `const` - constant
  """
  
  def __init__(self):
    OneDFit.__init__(self, ["A", "x0", "y0", "sigx", "sigy", "theta", "const"])
    self.setRootName("EllipGauss2dConst")

  def evaluate(self, co):
    """
      Evaluates the model for current parameter values.
      
      Parameters
      ----------
      co : array
           Specifies the points at which to evaluate the model.
    """
    if (self["sigx"] <= 0.0) or (self["sigy"] <= 0.0):
      raise(PE.PyAValError("Width(s) of Gaussian must be larger than zero.", \
                           solution="Set limits."))
    if self["theta"] > 2.*np.pi:
      raise(PE.PyAValError("The angle parameter must be 0 <= theta <= 2pi.", \
                           solution="Set limits."))
    a = np.cos(self['theta'])**2 / (2 * self['sigx']**2) + \
        np.sin(self['theta'])**2 / (2*self['sigy']**2)
    b = np.cos(2.*self['theta']) / (4 * self['sigx']**2) + \
        np.sin(2.*self['theta']) / (4 * self['sigy']**2)
    c = np.sin(self['theta'])**2 / (2 * self['sigx']**2) + \
        np.cos(self['theta'])**2 / (2 * self['sigy']**2)

    result = self["A"] * np.exp(\
      -1. * (a*(co[::,::,0]-self['x0'])**2 - 2*b*(co[::,::,0]-self['x0'])*(co[::,::,1]-self['y0']) + c*(co[::,::,1]-self['y0'])**2) \
      ) + self['const']
    return result
