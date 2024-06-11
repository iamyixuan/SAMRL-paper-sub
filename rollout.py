import pickle
import os
import argparse
import tensorflow as tf
from lambda_ppo.evaluate.eval_rollout import AgentEval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=35000)
    parser.add_argument('-name', type=str, default='test_rollout')
    cmd_args = parser.parse_args()


    MODEL_PATH = os.environ["MODEL_PATH"]
    # MODEL_PATH = "./Data/Bebop/Data/Data/MLP_PPO_KNL_learnedLag_21-04-17-2023-13-04-18/"
    with open(MODEL_PATH + "config", "rb") as file:
        config = pickle.load(file)
    args = config["args"]
    agent_config = config["agent"]
    neurons = args.neurons
    layers = args.layers
    env_steps = args.env_steps

    policy_config = {
        "action_dim": 2,
        "obs_dim": 16,
        "pi_layers": [neurons] * layers,
        "activation": tf.nn.tanh,
        "output_activation": None,
        "control_type": "continuous",
        "h_size": neurons,
        "batch_size": args.batch_size,
        "safety_layer": {"safety_level": 0},
    }

    evaluator = AgentEval(
        # sindy_path="./env_data/red_SINDYc_model_2022-4-15-downsample.joblib",
        policy_model_weigths=MODEL_PATH + "model_weights_epoch_300",
        policy_config=policy_config,
        seed=cmd_args.seed,
        env_steps=env_steps,
        lstm_actor="False",
        if_red="False",
        if_interp="False"
    )
    evaluator.rollout_trajectory(dir_path="./_rollout_DF/", file_name=f"{cmd_args.name}_" + f"{cmd_args.seed}")
