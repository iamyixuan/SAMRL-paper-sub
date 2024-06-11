"""
Class for random trajectory generation
"""
import numpy as np
import copy

from hyperopt import hp
import hyperopt.pyll.stochastic as hyps
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt


class TrajectorySampler:
    def __init__(self, argdict: dict, seed):
        """

        Parameters
        ----------
        argdict : dict of (str : )
            Each key represents a variable (e.g. v1 or v2).
            Value for each variable will have a dict:
                fn_name: str, function name (e.g., ddu_ramp)
                parameters: list of dict (
                    name: str, name of parameter (e.g. period/amplitude),
                    dist_type: str, distribution type (e.g. uniform),
                    par_value: tuple of float, values of parameters (e.g. min/max),
        """
        self.argdict = argdict
        self.rng = np.random.default_rng(seed)

    def sample_parameters(self):
        """
        Using the parameters defined for a particular variable,
        sample each of the parameters' distribution and add it
        to their dictionary.
        """
        for varname in self.argdict.keys():
            for parameter in self.argdict[varname]["parameters"]:
                sampler_bare = getattr(hp, parameter["dist_type"])
                sampler = sampler_bare(parameter["name"], *parameter["par_value"])
                parameter["sampled_value"] = hyps.sample(sampler, rng=self.rng)

    def sample_trajectory(self, timels: list):
        for varname in self.argdict.keys():
            trajls = []
            # Get alias of trajectory function
            fn = getattr(self, self.argdict[varname]["fn_name"])
            # Fill parameter dictionary from sample
            paramdict = {}
            for parameter in self.argdict[varname]["parameters"]:
                paramdict[parameter["name"]] = parameter["sampled_value"]
            for time in timels:
                trajls.append(fn(time, **paramdict))
            self.argdict[varname]["traj"] = trajls
            self.argdict[varname]["traj_time"] = list(timels)

        return copy.deepcopy(self.argdict)

    @staticmethod
    def ddu_ramp(time, **kwargs):
        """
        Down-to-Down-or-Up ramp
        This is to be used for generating load-follow sequences
        where power is the controlled variable.

        Parameters
        ----------
        time : float
            Time at which to ping fn
        p0 : float
            Starting value
        p1 : float
            Intermediate value
        p2 : float
            Final value
        rr1 : float
            Ramp rate of first down slope
        rr2 : float
            Ramp rate of second up/down slope
        rest1 : float
            Rest time after first down slope
        rest1 : float
            Rest time after second up/down slope
        minval : float
            Min saturation value of ramp (e.g., 192 MW for gFHR)
        maxval : float
            Max saturation value of ramp (e.g., 320 MW for gFHR)

                         rest1               rest2
        ───────────────┬──────────┬───────┬───────────┬──►
           AA <--P0    │          │       │           │
            AAA        │          │       │           │
              AA       │          │       │           │
               AAA     │          │     AAAAAAAAAAAAAA│ <--P2
          rr1    AA    │          │   AAA
                   AA  │          │  AA     rr2
                    AAA│          │AAA
         - - - - - - -AAAAAAAAAAAAA- <--P1 - - - - - - -
                                  AA
                                    A
                                    AAA     rr2
                                      AA
                                       AAA
                                         AAAAAAAAAAAAA  <--P2
        """
        p0 = kwargs["p0"]
        p1 = kwargs["p1"]
        p2 = kwargs["p2"]
        if p1 > kwargs["maxval"]:
            p1 = kwargs["maxval"]
        if p1 < kwargs["minval"]:
            p1 = kwargs["minval"]
        if p2 > kwargs["maxval"]:
            p2 = kwargs["maxval"]
        if p2 < kwargs["minval"]:
            p2 = kwargs["minval"]

        rr1 = kwargs["rr1"]
        rr2 = kwargs["rr2"]

        rest1 = kwargs["rest1"]
        rest2 = kwargs["rest2"]

        # Due to interpolation, we don't need to explicitly
        # define rr as negative or positive. It can be inferred
        # from values of p0, p1, and p2
        dt1 = abs((p1 - p0) / rr1)
        dt2 = abs((p2 - p1) / rr2)

        amps = np.array([p0, p1, p1, p2, p2, p2])
        times = np.array(
            [0, dt1, dt1 + rest1, dt1 + rest1 + dt2, dt1 + rest1 + dt2 + rest2, np.inf]
        )

        fn = interp1d(times, amps)

        return float(fn(time))


def pke_ddu_ramp_params(p1, p2, rr1, rr2):
    p_nom = 320.0e6
    # 10 minute rests
    rest1 = 10 * 60
    rest2 = 10 * 60
    temp_ls = [
        {
            "name": "p0",
            "dist_type": "choice",
            "par_value": (
                [
                    p_nom,
                ],
            ),
        },
        {"name": "p1", "dist_type": "uniform", "par_value": (p1 * 0.8, p1 * 1.2)},
        {"name": "p2", "dist_type": "uniform", "par_value": (p2 * 0.75, p2 * 1.25)},
        {"name": "rr1", "dist_type": "uniform", "par_value": (rr1 * 0.75, rr2)},
        {"name": "rr2", "dist_type": "uniform", "par_value": (rr2 * 0.75, rr2)},
        {
            "name": "rest1",
            "dist_type": "choice",
            "par_value": (
                [
                    rest1,
                ],
            ),
        },
        {
            "name": "rest2",
            "dist_type": "choice",
            "par_value": (
                [
                    rest2,
                ],
            ),
        },
        {
            "name": "minval",
            "dist_type": "choice",
            "par_value": (
                [
                    p_nom * 0.6,
                ],
            ),  # minumum
        },
        {
            "name": "maxval",
            "dist_type": "choice",
            "par_value": (
                [
                    p_nom,
                ],
            ),
        },
    ]
    return temp_ls


def get_traj(skip, seed):
    p1 = 256e6
    p2 = 256e6
    p_nom = 320.0e6  # maximum value
    rr_nom = p_nom * 0.5 / (10 * 60)

    scaler = lambda x: (x - 0.6 * p_nom) / (p_nom - 0.6 * p_nom) - 1

    argdict = {
        "power": {
            "fn_name": "ddu_ramp",
            "parameters": pke_ddu_ramp_params(p1, p2, rr_nom, rr_nom),
        }
    }
    TS = TrajectorySampler(argdict, seed=seed)
    TS.sample_parameters()  # run to sample parameter distributions defined above
    traj = TS.sample_trajectory(
        list(np.arange(0, 2251, 0.2))
    )  # run to get power trajectory
    traj = [scaler(t) for t in traj["power"]["traj"]]
    return np.asarray(traj)[::skip]


if __name__ == "__main__":
    traj = get_traj(5, 0)
    traj1 = get_traj(1, 0)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(np.arange(30), traj1[:30], color="r", linewidth=2, label="skip=1")
    ax.scatter(np.arange(30)[::5], traj1[:30][::5], s=50, label="skip=5", linewidth=3)
    # axs = ax.twiny()

    ax.set_box_aspect(1)
    ax.set_box_aspect(1)
    ax.legend(loc="center right")
    # axs.legend()
    plt.show()

    # print(traj[0], traj[1])
    # plt.plot(traj)
    # plt.show()
