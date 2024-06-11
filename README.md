# SAMRL

**Tip**: This repository only contains the model implementation and training script of the `lambda-ppo` algorithm
presented in the paper. The environment is under ongoing development and thus not publicly available.

## Running Training Scripts

### PPO Agent Training

To train a PPO agent with lagrangian primal-dual optimization, an LSTM actor, and an MLP-based cost critic to estimate the gradient of the lagrange multipliers:

```
mpirun -np 36 python train_ppo_agent.py -safety_level 0 -learn_cost_critic "True" -lstm_actor "True" -expid "PPO_LSTM_CC"
```

To train a PPO agent with lagrangian primal-dual optimization, an LSTM actor, and estimate the gradient of the lagrange multipliers directly from data:

```
mpirun -np 36 python train_ppo_agent.py -safety_level 0 -learn_cost_critic "False" -lstm_actor "True" -expid "PPO_LSTM_noCC"
```

To train a PPO agent with lagrangian primal-dual optimization, an MLP actor, and estimate the gradient of the lagrange multipliers directly from data:

```
mpirun -np 36 python train_ppo_agent.py -safety_level 0 -learn_cost_critic "False" -lstm_actor "False" -expid "PPO_noLSTM_noCC"
```

### SAC Agent Training

To train an SAC agent with lagrangian primal-dual optimization and an LSTM actor:

```
mpirun -np 36 python train_sac_agent.py -safety_level 0 -lstm_network "True" -expid "SAC_LSTM"
```

To train an SAC agent with lagrangian primal-dual optimization and an MLP actor:

```
mpirun -np 36 python train_sac_agent.py -safety_level 0 -lstm_network "False" -expid "SAC_noLSTM"
```

### Performance Comparison

![Performance Comparison](https://github.com/skhairy0/SAMRL/blob/main/Figs/perf_comp.jpg)
