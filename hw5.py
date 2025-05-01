#Problem 1
import numpy as np
import matplotlib.pyplot as plt


def f(x):
  return np.tanh(x)

def actualDerivative(x):
  return (1 / np.cosh(x)) ** 2

def forward_diff(f, value1, value2, pieces):
  epsilon = (value2-value1)/pieces
  derivatives = []
  values = np.linspace(value1, value2-epsilon, pieces)
  for val in values:
    x = (f(val+epsilon)-f(val))/epsilon
    derivatives.append(x)
  return derivatives, values

def central_diff(f, value1, value2, pieces):
  epsilon = (value2-value1)/pieces
  derivatives = []
  values = np.linspace(value1+epsilon, value2-epsilon, pieces)
  for val in values:
    x = (f(val+epsilon)-f(val-epsilon))/(2*epsilon)
    derivatives.append(x)
  return derivatives, values

def error(values):
  forwardErrors = []
  centralErrors = []
  for val in values:
    central, centralVals = central_diff(f, -2, 2, val)
    forward, forwardVals = forward_diff(f, -2, 2, val)

    actualCentral = actualDerivative(centralVals)
    actualFor = actualDerivative(forwardVals)

    centralError = np.abs((actualCentral - central)/actualCentral)
    forwardError = np.abs((actualFor - forward)/actualFor)

    centralErrors.append(np.mean(centralError))
    forwardErrors.append(np.mean(forwardError))

  return forwardErrors, centralErrors


values = np.logspace(2, 4, 10, dtype=int)
forwardErrors, centralErrors = error(values)

plt.loglog(values, forwardErrors, label = 'Forward')
plt.loglog(values, centralErrors, label = 'Central')
plt.xlabel('Pieces')
plt.ylabel('Error')
plt.title('Error vs Pieces')
plt.legend()
plt.show()
