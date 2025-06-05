import numpy as np
import math
from scipy.optimize import root
from scipy import special

class gnk():
  def __init__(self, sample_theta: bool = True, very_low: bool = False) -> None:
      self.sample_theta = sample_theta
      self.thete_default = np.array([3., 1., 2., math.exp(0.5)])   # same parameter as tutorial by pierrejacob
      self.param_range = np.array([[0.0, 0.0, 0.0, math.exp(0.5)], [3.0, 3.0, 3.0, 3.0]]) # when 0.0 < theta_4 < exp(0.5), numerical estimation of likelihood get very unstable.
      self.very_low = very_low

  def prior(self, n: int) -> np.ndarray:
      if not self.sample_theta:
         return np.repeat(np.expand_dims(self.thete_default, 0), n, 0)

      theta = np.random.rand(n, 4) * (self.param_range[1] - self.param_range[0]) + self.param_range[0]
     
      return theta
  
  def noise_generator(self, n: int, m: int) -> np.ndarray:
      U = np.random.rand(n, m, 1)
      return U
  
  def z_low(self, u: np.ndarray) -> np.ndarray:
      if self.very_low:
          z_u = np.sqrt(2) * self.erfinv_verylow(2 * u - 1)
      else:
          z_u = np.sqrt(2) * self.erfinv_low(2 * u - 1)
      return z_u.squeeze(-1)

  def z_high(self, u: np.ndarray) -> np.ndarray:
      z_u = np.sqrt(2) * special.erfinv(2 * u - 1)
      return z_u.squeeze(-1)

  def g_and_k(self, z_u: np.ndarray, theta: np.ndarray) -> np.ndarray:
      m = z_u.shape[-1] if len(z_u.shape) != 0 else 1
      if len(theta.shape) != 1:
          theta_1, theta_2, theta_3, theta_4 = (
            np.expand_dims(theta[:, 0], -1).repeat(m, axis = 1),  
            np.expand_dims(theta[:, 1], -1).repeat(m, axis = 1), 
            np.expand_dims(theta[:, 2], -1).repeat(m, axis = 1), 
            np.expand_dims(theta[:, 3], -1).repeat(m, axis = 1)
            )
      else:
          theta_1, theta_2, theta_3, theta_4 = theta[0], theta[1], theta[2], theta[3]
    
      a = (1 - np.exp(- theta_3 * z_u)) / (1 + np.exp(- theta_3 * z_u))
      b = (1 + z_u ** 2) ** (np.log(theta_4))
      x = theta_1 + theta_2 * (1 + 0.8 * a) * b * z_u
      return x
  
  def low_simulator(self, theta: np.ndarray, noise: np.ndarray) -> np.ndarray:
      z_u = self.z_low(noise)
      x = self.g_and_k(z_u = z_u, theta = theta)
      return x

  def high_simulator(self, theta: np.ndarray, noise: np.ndarray) -> np.ndarray:
      z_u = self.z_high(noise)
      x = self.g_and_k(z_u = z_u, theta = theta)
      return x
  
  def __call__(self, n: int, m: int, high: bool = True) -> np.ndarray:
      theta = self.prior(n)
      noise = self.noise_generator(n = n, m = m)
      if high:
          x = self.high_simulator(theta, noise)
      else:
          x = self.low_simulator(theta, noise)
      return theta, x
  
  def erfinv_low(self, u):
      return (np.pi / 2) * (u + (np.pi / 12) * u**3)
  
  def erfinv_verylow(self, u):
      # pi_ = 3
      return (np.pi / 2) * u
  
  def logprob(self, x_eval: np.ndarray, theta: np.ndarray, log: bool = True) -> np.ndarray:

      def cdf(x, theta):
          return np.vectorize(lambda xi: root(lambda u: xi - self.g_and_k(z_u = self.z_high(u), theta = theta), 0.5, tol = 1e-20, method = 'lm').x[0])(x)
      
      def numerical_gradient(func, x, theta, epsilon=1e-5):
          """
          Approximate the gradient of func at x using finite differences.
          """
          x_perturbed = x + epsilon
          grad = (func(x_perturbed, theta) - func(x, theta)) / epsilon
          
          return grad
      
      def pdf(x, theta):
          return np.vectorize(lambda xi: numerical_gradient(cdf, xi, theta))(x)
      
      _pdf = pdf(x_eval, theta)

      if log:
          return np.log(_pdf)
      
      else:
          return _pdf

    

