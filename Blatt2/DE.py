
import numpy as np
from scipy.optimize import differential_evolution

def ackley(x):
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    term1 = -20*np.exp(-0.2*np.sqrt(sum1/n))
    term2 = -np.exp(sum2/n)
    y = term1 + term2 + 20 + np.exp(1)
    return y

bounds = [(-32.768, 32.768)] * 20  # bounds for each dimension

result = differential_evolution(ackley, bounds, maxiter=1000, tol=1e-7, disp=True, popsize=1)
print("Minimum value found: ", result.fun)
print("Minimum location found: ", result.x)

# uses DE/best/1/bin
