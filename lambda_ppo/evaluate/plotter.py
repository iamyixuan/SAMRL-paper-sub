import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load


def penalized_reward(log):
    reward = log["ep_ret"][:, 0]
    cost1 = log["ep_ret"][:, 1]
    cost2 = log["ep_ret"][:, 2]
    lag1 = log["lagrange"][1:, 0]
    lag2 = log["lagrange"][1:, 1]

    # penalized_reward = reward - (lag1*(cost1 - 0.1*0.01) + lag2*(cost2 - 0.1*0.01))
    penalized_reward = reward - (cost1 + cost2)
    return penalized_reward


def read_logger(path):
    log_ = {}
    with open(path + "logger", "rb") as f:
        log = pickle.load(f)
        print(log.keys())
    for key in log.keys():
        log_[key] = np.array(log[key])
    return log_


def plotter_single(log1):
    """
    metrics in commom: pi_loss, ep_ret, ep_len, lag
    """
    plt.rcParams["font.size"] = 8
    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(1, 4)

    ax[0].plot(log1["ep_ret"][:, 0], color="k", label=r"$\mathcal{R}^{MLP}$")
    # ax[0].plot(log2["ep_ret"][:, 0], color="b", label=r"$\mathcal{R}^{LSTM}$")
    # ax[0].set_ylim(-500, 10)
    ax[0].set_ylabel(r"$\mathcal{R}(s_t, a_t)$")
    ax[0].legend(ncol=2, bbox_to_anchor=(0.5, 1.02), loc="lower center", fontsize=6)

    ax[1].plot(log1["ep_ret"][:, 1], color="blue", label=r"$\mathcal{C}_{in}^{MLP}$")
    # ax[1].plot(log2["ep_ret"][:, 1], color="b", label=r"$\mathcal{C}_{in}^{LSTM}$")
    ax[1].plot(
        log1["ep_ret"][:, 2],
        color="r",
        linestyle="--",
        label=r"$\mathcal{C}_{out}^{MLP}$",
    )
    # ax[1].plot(
    #     log2["ep_ret"][:, 2],
    #     color="b",
    #     linestyle="--",
    #     label=r"$\mathcal{C}_{out}^{LSTM}$",
    # )
    ax[1].legend(ncol=2, bbox_to_anchor=(0.5, 1.02), loc="lower center", fontsize=6)
    ax[1].set_ylabel(r"$\mathcal{C}$")

    ax[2].plot(log1["lagrange"][:, 0], color="blue", label=r"$\lambda_{in}^{MLP}$")
    # ax[2].plot(log2["lagrange"][:, 0], color="b", label=r"$\lambda_{in}^{LSTM}$")
    ax[2].plot(
        log1["lagrange"][:, 1],
        color="r",
        linestyle="--",
        label=r"$\lambda_{out}^{MLP}$",
    )
    # ax[2].plot(
    #     log2["lagrange"][:, 1],
    #     color="b",
    #     linestyle="--",
    #     label=r"$\lambda_{out}^{LSTM}$",
    # )
    ax[2].legend(ncol=2, bbox_to_anchor=(0.5, 1.02), loc="lower center", fontsize=6)
    ax[2].set_ylabel(r"$\lambda$")

    ax[3].plot(log1["entropy"], color="k", label=r"$H^{MLP}$")
    # ax[3].plot(log2["entropy"], color="b", label=r"$H^{LSTM}$")
    ax[3].set_ylabel(r"$H$")
    ax[3].legend(ncol=2, bbox_to_anchor=(0.5, 1.02), loc="lower center", fontsize=6)
    # ax[2,1].set_axis_off()
    for a in ax.flatten():
        # a.set_xlim(0, 600)
        a.grid(color="lightgray", linestyle="-.", linewidth=0.5)
        a.set_box_aspect(1)
        a.set_xlabel("Epochs")
        a.figure.set_size_inches(8, 8)
    fig.tight_layout()
    return fig


def plotter_compare(logList, nameList):
    """
    metrics in commom: pi_loss, ep_ret, ep_len, lag
    """
    plt.rcParams["font.size"] = 8
    plt.rcParams["lines.linewidth"] = 1.2
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["legend.fontsize"] = 5
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(1, 3)
    for i, log in enumerate(logList):
        N = 5
        p_reward = penalized_reward(log)
        filteredReward = np.convolve(p_reward, np.ones(N) / N, mode="valid")
        filteredCost1 = np.convolve(log["ep_ret"][:, 1], np.ones(N) / N, mode="valid")
        filteredCost2 = np.convolve(log["ep_ret"][:, 2], np.ones(N) / N, mode="valid")
        ax[0].plot(filteredReward, label=nameList[i])
        # ax[0].set_ylim(-500,10)
        ax[0].set_ylabel(r"$\mathcal{\bar{R}}(s_t, a_t)$")
        ax[0].set_xlim(0, 600)
        ax[1].plot(filteredCost1, label=nameList[i])
        ax[1].set_xlim(0, 600)
        ax[1].set_ylabel(r"$\mathcal{C}_{in}$")

        ax[2].plot(filteredCost2, label=nameList[i])
        ax[2].set_ylabel(r"$\mathcal{C}_{out}$")
        ax[2].set_xlim(0, 600)

    for a in ax.flatten():
        a.set_box_aspect(1)
        a.grid(linestyle="-.", linewidth="0.5", color="lightgray")
        a.set_xlabel("Epochs")
        # a.legend()
    handles, labels = ax[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, bbox_to_anchor=(0.5, 0.70), loc="upper center", ncol=6
    )
    legend.get_bbox_to_anchor().transformed(ax[1].transAxes)
    fig.tight_layout()
    return fig


def plot_compare_lags(model_list):
    from tmp.eval import AgentEval
    import re

    plt.rcParams["font.size"] = 14
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["font.family"] = "serif"

    policy_config = {
        "action_dim": 2,
        "obs_dim": 16,
        "pi_layers": [128, 128, 128],
        "activation": tf.nn.tanh,
        "output_activation": None,
        "control_type": "continuous",
        "h_size": 128,
        "batch_size": 25,  # [18,25]
        "safety_layer": {"safety_level": 0},
    }
    fig, axs = plt.subplots(1, 2)

    for j, model in enumerate(model_list):
        evaluator = AgentEval(
            traj_path="./utils/policy_sim/trajls_storage_DSQ.sql",
            sindy_path="./env_data/red_SINDYc_model_2022-4-15-downsample.joblib",
            policy_model_weigths="./Data/Bebop/Data/Data/MLP_PPO_KNL_learnedLag_21-04-17-2023-13-04-18/"
            + model,
            policy_config=policy_config,
            seed=30000 - 7,
            lstm_actor="False",
            if_red="False",
        )

        states, actions, requested_rho, rewards = evaluator.get_state_action(
            policy_config
        )  # add the demand following state as well.
        states_demand = evaluator.demand_follow_state()
        # Get time steps.
        ts = np.arange(0, len(states[:, 3]))
        # Inversely transform the states and bounds into their original scales and convert from K to C.
        in_temp = (
            evaluator.HX_s_tin_in_tran(states[:, 6])
            + evaluator.HX_s_tin_scaled_origin
            - 273.15
        )
        out_temp = (
            evaluator.HX_s_tout_in_tran(states[:, 7])
            + evaluator.HX_s_tout_scaled_origin
            - 273.15
        )
        lb_cons = (
            evaluator.HX_s_tin_in_tran(states[:, -2])
            + evaluator.HX_s_tin_scaled_origin
            - 273.15
        )
        ub_cons = (
            evaluator.HX_s_tout_in_tran(states[:, -1])
            + evaluator.HX_s_tout_scaled_origin
            - 273.15
        )

        # Get unconstrained states
        uc_in_temp = (
            evaluator.HX_s_tin_in_tran(states_demand[:, 6])
            + evaluator.HX_s_tin_scaled_origin
            - 273.15
        )
        uc_out_temp = (
            evaluator.HX_s_tout_in_tran(states_demand[:, 7])
            + evaluator.HX_s_tout_scaled_origin
            - 273.15
        )

        # set up colormap
        in_color = plt.cm.cool
        out_color = plt.cm.copper

        for i, ax in enumerate(axs):
            ax.set_box_aspect(1)
            if i == 0:
                if j == 0:
                    ax.plot(ts, lb_cons, color="black", label=r"$\mathcal{C}_{in}$")
                    ax.plot(
                        uc_in_temp, "--", color="blue", label=r"Load-following $T_{in}$"
                    )
                    ax.plot(
                        in_temp,
                        color=in_color(j / 4),
                        label="Epoch " + re.findall("[0-9]+", model)[0],
                    )
                    ax.set_xlabel("Timesteps")
                    ax.set_ylabel(r"$T_{in}$ ($^\circ C$)")
                    # ax.set_title('Inlet temperature')
                else:
                    ax.plot(
                        in_temp,
                        color=in_color(j / 4),
                        label="Epoch " + re.findall("[0-9]+", model)[0],
                    )

            else:
                if j == 0:
                    ax.plot(ts, ub_cons, color="black", label=r"$\mathcal{C}_{out}$")
                    ax.plot(
                        uc_out_temp, "--", color="r", label=r"Load-following $T_{out}$"
                    )
                    ax.plot(
                        out_temp,
                        color=out_color(j / 4),
                        label="Epoch " + re.findall("[0-9]+", model)[0],
                    )
                    ax.set_xlabel("Timesteps")
                    ax.set_ylabel(r"$T_{out}$ ($^\circ C$)")
                    # ax.set_title('Outlet temperature')
                else:
                    ax.plot(
                        out_temp,
                        color=out_color(j / 4),
                        label="Epoch " + re.findall("[0-9]+", model)[0],
                    )
    for a in axs:
        # a.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=8)
        a.set_box_aspect(1)
        a.figure.set_size_inches(7.5, 7.5)
        a.grid(color="lightgray", linestyle="-.", linewidth=0.5)

    fig.legend(
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.7),
        fontsize=9,
        bbox_transform=fig.transFigure,
    )
    fig.tight_layout()
    return fig


class PlotRollout:
    def __init__(self, df) -> None:
        scaler = load("../final_scalers/stateScaler_2023-05-15.joblib")
        plt.rcParams["lines.linewidth"] = 1.5
        plt.rcParams["font.size"] = 8
        plt.rcParams["legend.fontsize"] = 8
        plt.rcParams["axes.labelsize"] = 10

        self.in_temp = df["in_temp"]
        self.out_temp = df["out_temp"]
        self.core_flow = df["core_flow"]
        self.secondary_flow = df["secondary_flow"]
        self.pipe_inT = df["pipe110-inT"]
        self.lb_cons = df["lb"]
        self.ub_cons = df["ub"]
        self.uc_in_temp = df["uc_in_temp"]
        self.uc_out_temp = df["uc_out_temp"]
        self.requested_rho = df["demand"]
        self.actions = df["RL"]
        self.core_flow_d = df["core_flow_d"]
        self.secondary_flow_d = df["secondary_flow_d"]
        self.pipe_inT_d = df["pipe110-inT_d"]

        state_list = [self.core_flow, self.secondary_flow, self.pipe_inT] + 10 * [np.zeros_like(self.core_flow)]
        state_list_d = [self.core_flow_d, self.secondary_flow_d, self.pipe_inT_d] + 10 * [
            np.zeros_like(self.core_flow)
        ]
        self.rescaled_states = scaler.inverse_transform(np.array(state_list))
        self.rescaled_states_d = scaler.inverse_transform(np.array(state_list_d))

        if len(self.in_temp) > 2250:
            self.ts = np.arange(0, len(self.in_temp)) * 0.5 
        else:
            self.ts = np.arange(0, len(self.in_temp)) * 2.5 
    def plot_all_state(self):
        
        fig, axs = plt.subplots(2, 3, sharex=True, figsize=(8, 4))
        states = [
            (self.in_temp, self.uc_in_temp),
            (self.out_temp, self.uc_out_temp),
            (self.rescaled_states[0], self.rescaled_states_d[0]),
            (self.rescaled_states[1], self.rescaled_states_d[1]),
            (self.rescaled_states[2], self.rescaled_states_d[2]),
        ]
        labels = [
            r"$T_\mathrm{s,in}$ [$\degree$C]",
            r"$T_\mathrm{s,out}$ [$\degree$C]",
            r"$\dot{m}_\mathrm{p}$ [kg/s]",
            r"$\dot{m}_\mathrm{s}$ [kg/s]",
            r"$T_\mathrm{c,in}$ [$\degree$C]",
        ]
        for i, ax in enumerate(axs.ravel()):
            ax.set_box_aspect(1)
            ax.grid(color="lightgray", linestyle="-.", linewidth=0.4)
            if i >= 3:
                ax.set_xlabel(r"Time [$s$]")
            if i <= 4:
                ax.set_ylabel(labels[i])
                ax.plot(self.ts, states[i][0], color="k", label=r"$\lambda$-PPO")
                ax.plot(self.ts, states[i][1], "--", color="k", label="Load-following")
                # ax.set_ylim(states[i][1].min(), states[i][1].max()) # limit the plot limit for clearer comparison.
            if i == 0:
                ax.plot(self.ts, self.lb_cons, "--", color="red")
            elif i == 1:
                ax.plot(self.ts, self.ub_cons, "--", color="red")
            elif i == 5:
                ax.plot(
                    self.ts, self.actions, color="dodgerblue", linewidth=2, label=r"$\lambda$-PPO"
                )
                ax.plot(
                    self.ts,
                    self.requested_rho,
                    linestyle="dotted",
                    color="orangered",
                    label="Load demand",
                )
                ax.set_ylabel(r"$v_{\dot{Q}_\mathrm{RX}}$ [MW]")
                fig_a = plt.gca()
                fig_a.patch.set_facecolor("#E0E0E0")
                fig_a.patch.set_alpha(0.7)
        plt.tight_layout()
        return fig
    
    def plot_temp_act(self):
        fig, axs = plt.subplot(1, 3)
        for i, ax in enumerate(axs):
            ax.set_box_aspect(1)
            ax.figure.set_size_inches(8, 8)
            ax.grid(color="lightgray", linestyle="-.", linewidth=0.8)

            if i == 0:
                ax.plot(self.in_temp, color="blue", label=r"$T_{in}$")
                ax.plot(self.ts, self.lb_cons, color="black", label=r"$\mathcal{C}_{in}$")
                ax.plot(
                    self.uc_in_temp, "--", color="blue", label=r"Load-following $T_{in}$"
                )
                ax.set_xlabel(r"Time [$s$]")
                ax.set_ylabel(r"$T_{in}$ ($^\circ C$)")
                # ax.legend(bbox_to_anchor=((1.02, 1)), loc='upper right', ncol=1)
            elif i == 1:
                ax.plot(self.out_temp, color="r", label=r"$T_{out}$")
                ax.plot(self.ts, self.ub_cons, color="black", label=r"$\mathcal{C}_{out}$")
                ax.plot(self.uc_out_temp, "--", color="r", label=r"Load-following $T_{out}$")
                ax.set_xlabel(r"Time [$s$]")
                ax.set_ylabel(r"$T_{out}$ ($^\circ C$)")
                # ax.legend(bbox_to_anchor=((1.02, 1)), loc='upper right', ncol=1)
            else:
                ax.plot(self.requested_rho, "--", color="green", label="Load demand")
                ax.plot(self.actions, color="green", label="RL agant")
                ax.set_xlabel(r"Time [$s$]")
                ax.set_ylabel(r"$a_t$")
                # ax.legend(bbox_to_anchor=((1.02, 1)), loc='upper right', ncol=1)
        fig.tight_layout()
        fig.legend(
            bbox_to_anchor=(0.5, 0.63),
            ncol=4,
            loc="lower center",
            bbox_transform=fig.transFigure,
        )
        return fig





def plot_test(df):
    

    scaler = load("../final_scalers/stateScaler_2023-05-15.joblib")

    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["font.size"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.labelsize"] = 10

    in_temp = df["in_temp"]
    out_temp = df["out_temp"]
    core_flow = df["core_flow"]
    secondary_flow = df["secondary_flow"]
    pipe_inT = df["pipe110-inT"]
    lb_cons = df["lb"]
    ub_cons = df["ub"]
    uc_in_temp = df["uc_in_temp"]
    uc_out_temp = df["uc_out_temp"]
    requested_rho = df["demand"]
    actions = df["RL"]
    core_flow_d = df["core_flow_d"]
    secondary_flow_d = df["secondary_flow_d"]
    pipe_inT_d = df["pipe110-inT_d"]

    state_list = [core_flow, secondary_flow, pipe_inT] + 10 * [np.zeros_like(core_flow)]
    state_list_d = [core_flow_d, secondary_flow_d, pipe_inT_d] + 10 * [
        np.zeros_like(core_flow)
    ]
    rescaled_states = scaler.inverse_transform(np.array(state_list))
    rescaled_states_d = scaler.inverse_transform(np.array(state_list_d))

    ts = np.arange(0, len(in_temp)) * 0.5
    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(8, 4))
    states = [
        (in_temp, uc_in_temp),
        (out_temp, uc_out_temp),
        (rescaled_states[0], rescaled_states_d[0]),
        (rescaled_states[1], rescaled_states_d[1]),
        (rescaled_states[2], rescaled_states_d[2]),
    ]
    labels = [
        r"$T_\mathrm{s,in}$ [$\degree$C]",
        r"$T_\mathrm{s,out}$ [$\degree$C]",
        r"$\dot{m}_\mathrm{p}$ [kg/s]",
        r"$\dot{m}_\mathrm{s}$ [kg/s]",
        r"$T_\mathrm{c,in}$ [$\degree$C]",
    ]
    for i, ax in enumerate(axs.ravel()):
        ax.set_box_aspect(1)
        ax.grid(color="lightgray", linestyle="-.", linewidth=0.4)
        if i >= 3:
            ax.set_xlabel(r"Time [$s$]")
        if i <= 4:
            ax.set_ylabel(labels[i])
            ax.plot(ts, states[i][0], color="k", label=r"$\lambda$-PPO")
            ax.plot(ts, states[i][1], "--", color="k", label="Load-following")
            # ax.set_ylim(states[i][1].min(), states[i][1].max()) # limit the plot limit for clearer comparison.
        if i == 0:
            ax.plot(ts, lb_cons, "--", color="red")
        elif i == 1:
            ax.plot(ts, ub_cons, "--", color="red")
        elif i == 5:
            ax.plot(
                ts, actions, color="dodgerblue", linewidth=2, label=r"$\lambda$-PPO"
            )
            ax.plot(
                ts,
                requested_rho,
                linestyle="dotted",
                color="orangered",
                label="Load demand",
            )
            ax.set_ylabel(r"$v_{\dot{Q}_\mathrm{RX}}$ [MW]")
            fig_a = plt.gca()
            fig_a.patch.set_facecolor("#E0E0E0")
            fig_a.patch.set_alpha(0.7)
        # ax.legend(
        #             loc="lower center",
        #             ncol=2,
        #             bbox_to_anchor=(0.5, 1),
        #             fontsize=6,
        #             bbox_transform=ax.transAxes)

    # for i, ax in enumerate(axs):
    #     ax.set_box_aspect(1)
    #     ax.figure.set_size_inches(8, 8)
    #     ax.grid(color="lightgray", linestyle="-.", linewidth=0.8)

    #     if i == 0:
    #         ax.plot(in_temp, color="blue", label=r"$T_{in}$")
    #         ax.plot(ts, lb_cons, color="black", label=r"$\mathcal{C}_{in}$")
    #         ax.plot(
    #             uc_in_temp, "--", color="blue", label=r"Load-following $T_{in}$"
    #         )
    #         ax.set_xlabel("Timesteps")
    #         ax.set_ylabel(r"$T_{in}$ ($^\circ C$)")
    #         # ax.legend(bbox_to_anchor=((1.02, 1)), loc='upper right', ncol=1)
    #     elif i == 1:
    #         ax.plot(out_temp, color="r", label=r"$T_{out}$")
    #         ax.plot(ts, ub_cons, color="black", label=r"$\mathcal{C}_{out}$")
    #         ax.plot(uc_out_temp, "--", color="r", label=r"Load-following $T_{out}$")
    #         ax.set_xlabel("Timesteps")
    #         ax.set_ylabel(r"$T_{out}$ ($^\circ C$)")
    #         # ax.legend(bbox_to_anchor=((1.02, 1)), loc='upper right', ncol=1)
    #     else:
    #         ax.plot(requested_rho, "--", color="green", label="Load demand")
    #         ax.plot(actions, color="green", label="RL agant")
    #         ax.set_xlabel("Timesteps")
    #         ax.set_ylabel(r"$a_t$")
    #         # ax.legend(bbox_to_anchor=((1.02, 1)), loc='upper right', ncol=1)
    fig.tight_layout()
    # fig.legend(
    #     bbox_to_anchor=(0.5, 0.63),
    #     ncol=4,
    #     loc="lower center",
    #     bbox_transform=fig.transFigure,
    # )
    return fig




if __name__ == "__main__":
    # fig = plot_compare_lags(['model_weights_epoch_30', 'model_weights_epoch_80', 'model_weights_epoch_120', 'model_weights_epoch_300'])
    # plt.show()
    # fig.savefig('./eval_plots/paper_plots/varyingLagCompare.pdf', format='pdf', bbox_inches='tight')
    LSTMModel = read_logger(
        "./Data//Bebop//Data/Data/LSTM_PPO_KNL_learnedLag_27-04-17-2023-16-52-45/"
    )
    normalModel = read_logger(
        "./Data/Bebop/Data/Data/MLP_PPO_KNL_learnedLag_21-04-17-2023-13-04-18/"
    )
    lag0dot1 = read_logger(
        "./Data/Bebop/Data/Data/MLP_PPO_KNL_baseline_fixedLag0.1_79-04-04-2023-09-51-45/"
    )
    lag0dot2 = read_logger(
        "./Data/Bebop/Data/Data/MLP_PPO_KNL_baseline_fixedLag0.2_43-04-04-2023-09-51-45/"
    )
    lag0dot3 = read_logger(
        "./Data/Bebop/Data/Data/MLP_PPO_KNL_baseline_fixedLag0.3_18-04-13-2023-09-58-28/"
    )
    lag0dot4 = read_logger(
        "./Data/Bebop/Data/Data/MLP_PPO_KNL_baseline_fixedLag0.4_34-04-04-2023-09-51-45/"
    )
    lag0dot5 = read_logger(
        "./Data/Bebop/Data/Data/MLP_PPO_KNL_baseline_fixedLag0.5_52-04-13-2023-10-06-18/"
    )
    consSTDMLP = read_logger(
        "./Data/Bebop/Data/Data/MLP_PPO_KNL_consSTD0.3_58-04-23-2023-15-58-14/"
    )
    consSTDLSTM = read_logger(
        "./Data/Bebop/Data/Data/LSTM_PPO_KNL_consSTD0.3_41-04-23-2023-15-57-25/"
    )
    # lag0dot6 = read_logger('./Data/Bebop/Data/Data/MLP_PPO_KNL_baseline_fixedLag0.6_9-04-04-2023-09-51-45/')
    # lag0dot8 = read_logger('./Data/Bebop/Data/Data/MLP_PPO_KNL_baseline_fixedLag0.8_53-04-04-2023-09-51-45/')
    fig = plotter_single(consSTDMLP, consSTDLSTM)
    fig.savefig(
        "./eval_plots/paper_plots/agent_viz/lstm/MLPLSTMconsSTD.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    # log_list = [normalModel, lag0dot1, lag0dot2, lag0dot3, lag0dot4, lag0dot5]
    # name_list = [
    #     "SafePower-RL",
    #     r"$\lambda=0.1$",
    #     r"$\lambda=0.2$",
    #     r"$\lambda=0.3$",
    #     r"$\lambda=0.4$",
    #     r"$\lambda=0.5$",
    # ]
    # log_list_std = [normalModel, LSTMModel, consSTDMLP, consSTDLSTM]
    # name_list_std = ["MLP Actor", "LSTM Actor", "MLP Actor (const. std.)", "LSTM Actor (const. std.)"]
    # fig = plotter_compare(log_list_std, name_list_std)
    # fig.savefig(
    #     "./eval_plots/paper_plots/constantSTDcompare.pdf",
    #     bbox_inches="tight",
    # )
