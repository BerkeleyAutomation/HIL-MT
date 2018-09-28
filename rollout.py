import argparse
import os
import pickle
import time

import agents
import envs
from utils import DictTree


def rollout(config):
    env = envs.catalog(config.domain)
    agent = agents.catalog(DictTree(domain_name=config.domain, task_name=config.task, teacher=config.teacher, rollable=True, model_dirname=config.model))
    init_arg = env.reset(config.task)
    agent.reset(init_arg)
    trace = agent.rollout(env)
    try:
        os.makedirs("{}/{}".format(config.data, config.domain))
    except OSError:
        pass
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    pickle.dump(trace, open("{}/{}/{}.{}.pkl".format(config.data, config.domain, config.task, time_stamp), 'wb'), protocol=2)
    print("=== trace saved ===")
    raw_input("Press Enter to continue...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--model')
    parser.add_argument('--data')
    parser.add_argument('--teacher', action='store_true')
    args = parser.parse_args()
    rollout(args)
