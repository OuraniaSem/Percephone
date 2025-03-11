import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import boxcox, yeojohnson

# ====== sigmoid-related formulas ======
def minimal_k_bin(epsilon, delta_x):
    """
    Calculate the minimal steepness k for a logistic sigmoid function to transition
    from nearly 0 to nearly 1 over an interval of length delta_x with tolerance epsilon.

    The logistic function is defined as:
        f(x) = 1 / (1 + exp(-k*(x - delta_x/2)))
    where f(0) <= epsilon and f(delta_x) >= 1 - epsilon.

    Parameters:
        epsilon (float): The tolerance (at each side, to cumulate) level for f(0) and f(delta_x).
        delta_x (float): The x-interval over which the transition occurs.

    Returns:
        float: The minimal k value satisfying the tolerance constraints.
    """
    return (2 / delta_x) * math.log((1 / epsilon) - 1)


def minimal_k_amp(p):
    """
    Calculate the minimal k value for a logistic sigmoid function
    f(x) = 1 / (1 + exp(-k*(x-0.5))) such that the difference in the
    function's output between x=0 and x=1 is exactly p.

    That is, we require:
        f(1) - f(0) = p,
    where 0 < p < 1.

    Parameters:
        p (float): The desired absolute change in the y-value between x=0 and x=1.

    Returns:
        float: The minimal k value.
    """
    # k = 2 * ln((1+p)/(1-p))
    return 2 * math.log((1 + p) / (1 - p))


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y


def sigmoid_Hill(x, n, k):
    y = (x ** n) / (x ** n + k ** n)
    # y = 1/(1+(k/x)**n)
    return y


def sigmoid_fit(xdata, ydata):
    popt, pcov = curve_fit(sigmoid, xdata, ydata, maxfev=100000)
    fix_value = xdata[-1]  # + 1
    # slope, intercept = np.polyfit(x, y, 1)
    # slope = float("{:.2f}".format(slope))
    # Get r2 score
    xdata = xdata.astype(float)
    # y_pred = sigmoid(xdata, *popt)
    # r2 = r2_score(ydata, y_pred)
    # liste_r2.append(r2)
    x0 = popt[0]
    k = popt[1]
    x = np.linspace(0, fix_value, 50)
    y = sigmoid(x, x0, k)
    return x, y, x0, k


# ====== trasnformation functions ======
def inv_transform(x):
    return 1 / x


def arcsin_transform(x):
    # Use a consistent transformation based on the maximum value
    if np.nanmax(x) > 1:
        return np.arcsin(np.sqrt(x / 100))
    else:
        return np.arcsin(np.sqrt(x))

