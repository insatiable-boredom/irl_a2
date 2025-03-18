import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import unittest


# Create a plot that will be used later in the functions to plot graphs, once per experiment
def create_plot():
    """
    Will need to create subplots or axes object from matplotlib.
    Should receive:
    title, x_label, y_label.
    Can receive:
    domain, x_ticks, y_ticks.
    """
    raise NotImplementedError("Function not yet created.")


# Smooth a graph curve to be displayed in the plot, use when appropriate only
def smooth_curve():
    """
    Will need to smooth a curve of points using savgol filter.
    Should receive:
    original curve points vector, smoothing window.
    Can receive:
    poly.
    Should return the smoothed curve points vector.
    """
    raise NotImplementedError("Function not yet created.")


# Add a graph, curve and legend, to the experiment plot
def add_graph():
    """
    Will need to smooth the curve of a graph if necessary and then add it with a legend to the plot.
    Should receive:
    plot, original graph, legend label.
    Can receive:
    color, line type.
    """
    raise NotImplementedError("Implement create_plot and smooth_curve first.")


# Repeat a training and result iteration, generate the graph
def run_repetitions():
    """
    Will need to run a number of repetitions that reset the agent and average out the results.
    Should receive:
    agent, environment, number of repetitions.
    Should return an averaged results graph.
    """
    raise NotImplementedError("Function not yet created.")


# Initialize the agent and environment, call the other functions
def experiment():
    """
    Will need iterate over multiple agents and environments to initialize then run experiment, create plot then display.
    Should receive:
    all agents, all environments, number of repetitions.
    Can receive:
    customization settings.
    """
    raise NotImplementedError("Complete all prior functions then work here.")


if __name__ == "__main__":
    print("Wait you must, young jedi.")
    unittest.main()


# CREATE BASIC UNIT TESTS
class TestExperimentalFunctions(unittest.TestCase):
    """
    Test suit to ensure correct types and some example values for the functions used in the experiments.
    """

    def test_create_plot(self):
        pass

    def test_smooth_curve(self):
        pass

    def test_add_graph(self):
        pass

    def test_run_repetitions(self):
        pass

    def test_experiment(self):
        pass
