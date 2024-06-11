import pandas as pd
import re
from lambda_ppo.evaluate.plotter import PlotRollout
from lambda_ppo.evaluate.extraStatePlot import CustomScaler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-df_path', type=str, default='./_rollout_DF/MLP_skip5_35000.csv')

    args = parser.parse_args()
    name = args.df_path.split('/')[-1].replace('.csv', '')
    df_path = args.df_path
    df = pd.read_csv(df_path)

    plotter = PlotRollout(df)
    fig = plotter.plot_all_state()
    fig.savefig(f'./_rollout_Fig/{name}.pdf', format='pdf')

   

    # if args.train == "True":
    #     log = read_logger(args.log_path)
    #     fig = plotter_single(log)
    #     fig.savefig(f'./{args.name}.pdf', format='pdf')
    # else:
    #     fig = plot_test(df)
    #     fig.savefig(f'./{args.name}.pdf', format='pdf')

    
    # import pickle
    # logger = read_logger('./Data/MLPskip1-2843848_84-06-14-2023-12-31-50/') 
    # fig = plotter_single(logger)
    