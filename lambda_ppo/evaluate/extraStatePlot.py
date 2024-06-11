import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plotter import plot_test

from sklearn.base import BaseEstimator, TransformerMixin


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, dataMin, dataMax, constant):
        self.dataMin = dataMin.reshape(-1, 1)
        self.constant = constant.reshape(-1, 1)
        self.dataMax = dataMax.reshape(-1, 1)
        self.scaleMin = 0.1
        self.scaleMax = 0.9

    def transform(self, X, y=None):
        return (
            ((X - self.dataMin) / (self.dataMax - self.dataMin))
            * (self.scaleMax - self.scaleMin)
            + self.scaleMin
            - self.constant
        )

    def inverse_transform(self, X, y=None):
        return (
            (X + self.constant.reshape(-1, 1) - self.scaleMin)
            / (self.scaleMax - self.scaleMin)
        ) * (
            self.dataMax.reshape(-1, 1) - self.dataMin.reshape(-1, 1)
        ) + self.dataMin.reshape(
            -1, 1
        )


def main():
    df = pd.read_csv(
        "./eval_plots/paper_plots/agent_viz/test/extraStates/testTraj0.csv"
    )
    fig = plot_test(df)
    fig.savefig(
        "./eval_plots/paper_plots/agent_viz/presentation_plots_data/tmp_state_vis.pdf"
    )


if __name__ == "__main__":
    main()
