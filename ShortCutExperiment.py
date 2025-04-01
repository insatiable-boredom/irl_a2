import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent, nStepSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment


def create_plot(title="Experiment", x_label="Parameter setting", y_label="Average reward") -> plt.axes:
    """
    Create basic axes plot to overlay graphs, all created plots show as separate figures.

    :param title: Figures title.
    :param x_label: X-axis label.
    :param y_label: Y-axis label.

    :return: Matplotlib axes object containing the relevant plot figure.
    """

    plot = plt.axes()
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    return plot


def smooth_curve(curve: np.typing.ArrayLike, window=3, poly=1) -> np.typing.ArrayLike:
    """
    Basic smoothing function for vector of curve points, preferably use only when necessary.

    :param curve: Array like ordered collection of points.
    :param window: Number of co-efficients, must be less than length of curve points.
    :param poly: Order of polynomial of co-efficients.

    :return: Smoothed curve points.
    """

    return savgol_filter(curve, window, poly)


# Add a graph, curve and legend, to the experiment plot
def add_graph(plot: plt.Axes, curve: np.typing.ArrayLike, label: str = None, smooth: bool = False) -> None:
    """
    Will need to smooth the curve of a graph if necessary and then add it with a legend to the plot.
    Should receive:
    plot, original graph, legend label.
    Can receive:
    color, line type.
    """
    points = smooth_curve(curve) if smooth else curve
    if label is None:
        plot.plot(points)
    else:
        plot.plot(points, label=label)
        plot.legend()


# Repeat a training and result iteration, generate the graph
def run_repetitions(agent, seed: int = None, n_repetitions: int = 500, n_episodes: int = 1000):
    """
    Will need to run a number of repetitions that reset the agent and average out the results.
    Should receive:
    agent, environment, number of repetitions.
    Should return an averaged results graph.
    """

    np.random.seed(seed)  # For reproducibility or randomness

    rewards = np.zeros((n_repetitions, n_episodes))
    for rep in range(n_repetitions):
        rewards[rep] = agent.train(n_episodes)

    average_rewards = np.mean(rewards, axis=0)
    return np.mean(average_rewards)


# Initialize the agent and environment, call the other functions
def experiment(environments, agents, reps, eps, seed):
    """
    Will need iterate over multiple agents and environments to initialize then run experiment, create plot then display.
    Should receive:
    all agents, all environments, number of repetitions.
    Can receive:
    customization settings.
    """
    for env in environments:
        for agent in agents:
            e_plot = create_plot(f"Agent: {agent.__name__}, Environment: {env.__name__}")

    raise NotImplementedError("Complete all prior functions then work here.")


if __name__ == "__main__":
    print("Wait you must, young jedi.")
    
