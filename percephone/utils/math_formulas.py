import math


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