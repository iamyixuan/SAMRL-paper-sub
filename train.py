import os
import argparse
from lambda_ppo.train_ppo import train

if __name__ == "__main__":

    os.environ["BOOTSTRAP_FOR_LONG_EPISODE"] = 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lam_lr",
        type=float,
        default=0.002,
        help="Learning rate for lagrange multiplier",
    )
    parser.add_argument(
        "-entrop_coeff", type=float, default=0.05, help="entropy coefficient"
    )
    parser.add_argument(
        "-target_kl",
        type=float,
        default=0.01,
        help="target kl divergence for early stopping",
    )
    parser.add_argument("-delta", type=float, default=0.1, help="delta")
    parser.add_argument("-gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "-lam_init",
        type=float,
        default=0.0,
        help="Initial value for lagrange multiplier",
    )
    parser.add_argument(
        "-histlen",
        type=int,
        default=1,
        help="Number of historical timesteps in state vector",
    )
    parser.add_argument(
        "-savefreq",
        type=int,
        default=1,
        help="Save policy network weights every this number of epochs",
    )
    parser.add_argument(
        "-env_steps", type=int, default=2250, help="number of interactions per epoch"
    )  # 20000 for full env
    parser.add_argument(
        "-epochs", type=int, default=400, help="number of training epochs"
    )
    parser.add_argument(
        "-batch_size", type=int, default=25, help="lstm batch size"
    )  # should always be 20 x 100 (trace length), ghfr 18*125
    parser.add_argument(
        "-neurons", type=int, default=128, help="number of neurons per layer"
    )
    parser.add_argument("-layers", type=int, default=3, help="number of layers")
    parser.add_argument("-expid", type=str, default="test", help="Problem type")
    parser.add_argument(
        "-learn_cost_critic",
        type=str,
        choices=("True", "False"),
        default="True",
        help="whether to learn a cost critic or not",
    )
    parser.add_argument(
        "-lstm_actor",
        type=str,
        choices=("True", "False"),
        default="False",
        help="whether to use an lstm actor or not",
    )
    parser.add_argument(
        "-safety_level", type=int, default=0, help="chosen safety method"
    )
    parser.add_argument(
        "-const_std",
        type=str,
        default="False",
        help="set constant policy standard deviation",
    )
    parser.add_argument(
        "-std_lb",
        type=float,
        default=0.001,
        help="The lower bound of policy standard deviation to ensure the output standard deviation is not smaller than 0",
    )

    parser.add_argument(
        "-limiter", type=str, choices=("True", "False"), default="False"
    )  # use the action limiter with MLP actor.
    parser.add_argument(
        "-reduced_dim", type=str, choices=("True", "False"), default="False"
    )
    parser.add_argument("-saved_model_path", type=str, default='./Data/2862549-MLPskip1-subEpUpdate-Bootstrap_21-06-26-2023-10-52-14/')


    args = parser.parse_args()
    args.learn_cost_critic = args.learn_cost_critic == "True"
    args.lstm_actor = args.lstm_actor == "True"

    train(args, load_weights=True)