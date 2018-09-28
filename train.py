import argparse
import glob
import itertools
import os
import pickle
import time

import numpy as np

import agents
from skillhub import client
from utils import DictTree


def load_traces(dirname):
    return [pickle.load(open(filename, 'rb')) for filename in glob.iglob("{}/*.pkl".format(dirname))]


def _train(data_dirname, new_agent, past_agents, config):
    """

    Args:
        config (DictTree)
    """
    shared_skills = {}
    for past_agent in past_agents:
        for skill_name in past_agent.skillset:
            shared_skills.setdefault(skill_name, []).append(DictTree(
                agent_name=past_agent.task_name,
                skill_name=skill_name,
            ))
    traces = load_traces("{}/{}".format(data_dirname, new_agent.task_name))
    print("Training on {} traces".format(len(traces)))
    np.random.shuffle(traces)
    return client.train_agent(new_agent, traces, config | DictTree(shared_skills=shared_skills))


def train(config):
    all_agents = [agents.catalog(DictTree(domain_name=config.domain, task_name=task_name, rollable=False, teacher=False)) for task_name in config.tasks]
    data_dirname = "{}/{}".format(config.data, config.domain)
    for agent in all_agents[:-1]:
        client.delete(agent)
        _train(data_dirname, agent, [], DictTree(
            modes=['independent'], batch_size=None, validate=False,
            model_dirname="model/{}/{}".format(config.domain, agent.task_name)))
    results = DictTree()
    if config.independent:
        modes_list = [['independent']]
    else:
        modes_list = itertools.product(['validation', ''], ['training', ''], ['independent', ''])
    for modes in modes_list:
        modes = [mode for mode in modes if mode]
        if not modes:
            continue
        print("Training with modes: {}".format(', '.join(modes)))
        client.delete(all_agents[-1])
        results['+'.join(modes)] = _train(data_dirname, all_agents[-1], all_agents[:-1], DictTree(
            modes=modes, batch_size=(None if config.full_batch else 1), validate=True,
            model_dirname="model/{}/{}_{}".format(config.domain, ".".join(config.tasks), "+".join(modes))))
    try:
        os.makedirs("results/{}/{}".format(config.domain, ".".join(config.tasks)))
    except OSError:
        pass
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    pickle.dump(results, open("results/{}/{}/{}.{}.pkl".format(config.domain, ".".join(config.tasks), all_agents[-1].task_name, time_stamp), 'wb'), protocol=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--runs')
    parser.add_argument('--independent', action='store_true')
    parser.add_argument('--full-batch', action='store_true')
    args = parser.parse_args()
    for _ in range(int(args.runs) or 1):
        train(args)
