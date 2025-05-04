#My GIT Comment

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


#Problem 2 a.

import numpy as np
import matplotlib.pyplot as plt


def f(x):
  return np.tanh(x)

def actualDerivative(x):
  return (1 / np.cosh(x)) ** 2

def Secondforward_diff(f, value1, value2, pieces):
  epsilon = (value2-value1)/pieces
  derivatives = []
  values = np.linspace(value1, value2-2*epsilon, pieces)
  for val in values:
    x = (4*f(val+epsilon)-f(val+2*epsilon)-3*f(val))/(2*epsilon)
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
  SecondForwardErrors = []
  centralErrors = []
  for val in values:
    central, centralVals = central_diff(f, -2, 2, val)
    forward, forwardVals = Secondforward_diff(f, -2, 2, val)

    actualCentral = actualDerivative(centralVals)
    actualFor = actualDerivative(forwardVals)

    centralError = np.abs((actualCentral - central)/actualCentral)
    forwardError = np.abs((actualFor - forward)/actualFor)

    centralErrors.append(np.mean(centralError))
    SecondForwardErrors.append(np.mean(forwardError))

  return SecondForwardErrors, centralErrors


values = np.logspace(2, 4, 10, dtype=int)
SecondforwardErrors, centralErrors = error(values)

plt.loglog(values, SecondforwardErrors, label = 'Second Forward')
plt.loglog(values, centralErrors, label = 'Central')
plt.xlabel('Pieces')
plt.ylabel('Error')
plt.title('Error vs Pieces')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt


def f(x):
  return np.tanh(x)

def actualDerivative(x):
  return (1 / np.cosh(x)) ** 2


def central_diff(f, value1, value2, pieces):
  epsilon = (value2-value1)/pieces
  derivatives = []
  values = np.linspace(value1, value2-epsilon, pieces)
  for val in values:
    if val == value1:
      x = (4*f(val+epsilon)-f(val+2*epsilon)-3*f(val))/(2*epsilon)
      derivatives.append(x)
    else:
      x = (f(val+epsilon)-f(val-epsilon))/(2*epsilon)
      derivatives.append(x)
  return derivatives, values



def error(value):

  central, centralVals = central_diff(f, -2, 2, value)

  actualCentral = actualDerivative(centralVals)

  centralError = np.abs((actualCentral - central)/actualCentral)

  return centralError



def definite(values):
  Errors = []
  xs = []
  for value in values:
    Error = error(value)
    Errors.append(Error)
    x = np.linspace(-2, 2, value)
    xs.append(x)
  return Errors, xs
values = [50, 100, 200]


Errors, xs = definite(values)

for value in range(len(Errors)):
  plt.plot(xs[value], Errors[value], label = f'{values[value]}')

plt.xlabel('X')
plt.ylabel('Error')
plt.title('Error vs X-value')
plt.legend()
plt.show()



#Extra Credit

import numpy as np
import matplotlib.pyplot as plt


def f(x):
  return np.tanh(x)

def actualDerivative(x):
  return (1 / np.cosh(x)) ** 2


def central_diff(f, value1, value2, pieces):
  epsilon = (value2-value1)/pieces
  derivatives = []
  values = np.linspace(value1, value2, pieces)
  for val in values:
    if val == value1:
      x = (4*f(val+epsilon)-f(val+2*epsilon)-3*f(val))/(2*epsilon)
      derivatives.append(x)
    elif val == value2:
      x = (3*f(val)- 4*f(val-epsilon)+f(val-2*epsilon))/(2*epsilon)
      derivatives.append(x)
    else:
      x = (f(val+epsilon)-f(val-epsilon))/(2*epsilon)
      derivatives.append(x)
  return derivatives, values



def error(value):
  central, centralVals = central_diff(f, -2, 2, value)

  actualCentral = actualDerivative(centralVals)

  centralError = np.abs((actualCentral - central)/actualCentral)



  return centralError



def definite(values):
  Errors = []
  xs = []
  for value in values:
    Error = error(value)
    Errors.append(Error)
    x = np.linspace(-2, 2, value)
    xs.append(x)
  return Errors, xs
values = [50, 100, 200]


Errors, xs = definite(values)


for value in range(len(Errors)):
  plt.plot(xs[value], Errors[value], label = f'{values[value]}')

plt.xlabel('X')
plt.ylabel('Error')
plt.title('Error vs X-value')
plt.legend()
plt.show()
