"""
sam-dc/smg.py
Akshay J. Dave
Argonne National Laboratory
"""

import numpy as np
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import MinMaxScaler as MMS
from sqlitedict import SqliteDict as SQD


def load_key(key, file_nm):
    """
    Returns value of key in sqlfile file_nm.
    """
    try:
        with SQD(file_nm) as mydict:
            value = mydict[key]
    except Exception as ex:
        print("Error during loading data:", ex)
    return value


class TrajectoryServer:
    def __init__(self, sql_nm, key_ls, json_flag=False):
        self._load_sql_db(sql_nm, key_ls, json_flag)

    def _load_sql_db(self, sql_nm: str, key_ls: list, json_flag: bool):
        # Load each key as separate trajs list
        val_ls = []
        for key in key_ls:
            if json_flag:
                val_ls.append(load_key_json(key, sql_nm))
            else:
                val_ls.append(load_key(key, sql_nm))
        self.trajls = [entry for traj in val_ls for entry in traj]

        # Check if PID data existed
        pidDataSetNames = set()
        pidDataSetSize = {}
        for traj in self.trajls:
            if "pid_comp_ls" in traj["results"]:
                for entry in list(traj["results"]["pid_comp_ls"].keys()):
                    pidDataSetNames.add(entry)
                    sz = np.array(traj["results"]["pid_comp_ls"][entry]).shape
                    pidDataSetSize[entry] = sz
        # Set order MAY changed during execution
        # Sorting ensures same order is received.
        pidDataSetNames = sorted(pidDataSetNames)
        self.pidDataSetNames = [sNm + "_REF" for sNm in pidDataSetNames]

        # Aggregate state, actuator, and time data from each traj
        state_agg = []
        act_agg = []
        time_agg = []
        pid_comp_agg = []
        pid_setpoint_agg = []

        for traj in self.trajls:
            state_agg.append(np.array(traj["results"]["state_ls"]))
            act_agg.append(np.array(traj["results"]["act_ls"]))
            time_agg.append(np.array(traj["results"]["time_ls"]))
            # If PID data exists across datasets
            if len(pidDataSetNames) > 0:
                pid_comp = []
                pid_val = []
                for pidName in pidDataSetNames:
                    if pidName in traj["results"]["pid_comp_ls"]:
                        pid_comp.append(traj["results"]["pid_comp_ls"][pidName])
                        pid_val.append(traj["results"]["pid_setpoint_ls"][pidName])
                    else:
                        pid_comp.append(np.zeros(pidDataSetSize[pidName]))
                        pid_val.append(np.zeros(pidDataSetSize[pidName][0]))
                pid_comp_agg.append(np.hstack(pid_comp))
                pid_setpoint_agg.append(np.column_stack(pid_val))

        self.states = np.array(state_agg)
        self.acts = np.array(act_agg)
        self.times = np.array(time_agg)
        self.pidComps = np.array(pid_comp_agg)
        self.pidSetpts = np.array(pid_setpoint_agg)

        # Prepare names for all data
        self.stateNames = self.trajls[0]["results"]["state_nm"]
        self.actNames = self.trajls[0]["results"]["act_nm"]
        # Order of names will be preseved
        pidCompNames = []
        for pidCompName in pidDataSetNames:
            pidCompNames.append("P+" + pidCompName)
            pidCompNames.append("I+" + pidCompName)
            pidCompNames.append("D+" + pidCompName)
        self.pidCompNames = pidCompNames

        # Setup MMS scalers for state and actuators
        self.stateMMS = MMS((0.1, 0.9))
        self.actMMS = MMS((0.1, 0.9))
        self.pidCompMMS = MMS((0.1, 0.9))
        self.pidSetptsMMS = MMS((0.1, 0.9))
        self.stateMMS.fit(self.states.reshape(-1, self.states.shape[-1]))
        self.actMMS.fit(self.acts.reshape(-1, self.acts.shape[-1]))
        self.pidCompMMS.fit(self.pidComps.reshape(-1, self.pidComps.shape[-1]))
        self.pidSetptsMMS.fit(self.pidSetpts.reshape(-1, self.pidSetpts.shape[-1]))

        # Split the data into training and test batches
        if self.states.shape[0] > 1:
            (
                act_tr,
                act_te,
                state_tr,
                state_te,
                pidComp_tr,
                pidComp_te,
                pidSetpt_tr,
                pidSetpt_te,
                time_tr,
                time_te,
            ) = TTS(
                self.acts,
                self.states,
                self.pidComps,
                self.pidSetpts,
                self.times,
                test_size=0.2,
                random_state=42,
            )
            self.act_tr = act_tr
            self.act_te = act_te
            self.state_tr = state_tr
            self.state_te = state_te
            self.pidComp_tr = pidComp_tr
            self.pidComp_te = pidComp_te
            self.pidSetpt_tr = pidSetpt_tr
            self.pidSetpt_te = pidSetpt_te
            self.time_tr = time_tr
            self.time_te = time_te

            self.s_act_tr = TrajectoryServer.batch_scaler(self.actMMS.transform, act_tr)
            self.s_act_te = TrajectoryServer.batch_scaler(self.actMMS.transform, act_te)
            self.s_state_tr = TrajectoryServer.batch_scaler(
                self.stateMMS.transform, state_tr
            )
            self.s_state_te = TrajectoryServer.batch_scaler(
                self.stateMMS.transform, state_te
            )
            self.s_pidComp_tr = TrajectoryServer.batch_scaler(
                self.pidCompMMS.transform, pidComp_tr
            )
            self.s_pidComp_te = TrajectoryServer.batch_scaler(
                self.pidCompMMS.transform, pidComp_te
            )
            self.s_pidSetpt_tr = TrajectoryServer.batch_scaler(
                self.pidSetptsMMS.transform, pidSetpt_tr
            )
            self.s_pidSetpt_te = TrajectoryServer.batch_scaler(
                self.pidSetptsMMS.transform, pidSetpt_te
            )
        else:
            # Only 1 dataset
            self.act_tr = self.acts
            self.state_tr = self.states
            self.pidComp_tr = self.pidComps
            self.pidSetpt_tr = self.pidSetpts
            self.time_tr = self.times

            self.s_act_tr = TrajectoryServer.batch_scaler(
                self.actMMS.transform, self.act_tr
            )
            self.s_state_tr = TrajectoryServer.batch_scaler(
                self.stateMMS.transform, self.state_tr
            )
            self.s_pidComp_tr = TrajectoryServer.batch_scaler(
                self.pidCompMMS.transform, self.pidComp_tr
            )
            self.s_pidSetpt_tr = TrajectoryServer.batch_scaler(
                self.pidSetptsMMS.transform, self.pidSetpt_tr
            )

            self.act_te = None
            self.state_te = None
            self.pidComp_te = None
            self.pidSetpt_te = None
            self.time_te = None
            self.s_act_te = None
            self.s_state_te = None
            self.s_pidComp_te = None
            self.s_pidSetpt_te = None

    @staticmethod
    def batch_scaler(transform_fn, inp_batch):
        out_batch = np.copy(inp_batch)
        for idx in range(inp_batch.shape[0]):
            out_batch[idx] = transform_fn(inp_batch[idx])
        return out_batch

    @property
    def scaled_states(self):
        return self.s_state_tr, self.s_state_te

    @property
    def scaled_acts(self):
        return self.s_act_tr, self.s_act_te

    @property
    def time(self):
        return self.time_tr, self.time_te
