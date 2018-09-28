# Multi-Task Hierarchical Imitation Learning of Robot Skills

Code for running the *SkillHub* server, generating annotated demonstrations, learning hierarchical controllers, and rolloing learned controllers on an HSR robot.

## Installation

In addition to the dependencies in [requirements.txt](requirements.txt), the [`HSREnv`](envs/hsr.py#L202) depends on [pyyolo](https://github.com/digitalbrain79/pyyolo).
Please follow [these instructions](https://github.com/digitalbrain79/pyyolo#building) to install pyyolo, and then change the paths in [vision.py](envs/vision.py) to your installation path.

## Running SkillHub

To run the SkillHub server:

`python skillhub/server.py`

To run the server in debug mode, set the `DEBUG` flag to `True` in [server.py](skillhub/server.py#L21).

## Providing demonstrations

To rollout a demonstration of the `SetTable` task:

`python rollout.py --domain dishes --task SetTable --data data --teacher`

Available tasks are: `ClearTable` and `SetTable` in the `dishes` domain; and `Pyramid<n>` (pyramid of height `n`) in the `pyramid` domain.
Annotated demonstrations will be saved in the path provided to `--data`.

To implement new domains, inherit from [`Env`](envs/env.py#L5) (or its subclass [`HSREnv`](envs/hsr.py#L202)) -- see [`PyramidEnv`](envs/pyramid.py#L4) for example.
To implement new tasks, inherit from [`Agent`](agents/agent.py#L6) (or its subclass [`HierarchicalAgent`](agents/hierarchical.py#L8)) -- see [`PyramidAgent`](agents/pyramid.py#L229) for example.

Human teleoperation is performed by actions starting with `Record_`, which pause the script to allow human control and record it.
Before training, the recorded demonstrations must be fixed to format the recorded control as robot control.
To fix all demonstrations in a path:

`python -c "import fix; fix.fix_record('data/dishes/SetTable_fix', 'data/dishes/SetTable_rec')"`

## Training hierarchical controller

To train controllers for the `ClearTable` and `SetTable` tasks:

`python train.py --domain dishes --tasks ClearTable SetTable --data data`

All controllers but the last will be trained independently from all their available data.
The last controller (`SetTable` in this case) will be trained with *detailed mode selection* with the data of past tasks and with new demonstrations added one by one.
The results of training will be saved in `results/dishes/ClearTable.SetTable`.

Use `--runs <n>` to repeat training for `n` independent trials.
Use `--independent` to train the last controller only independently of the other tasks.
Use `--full-batch` to train the last controller from all available data, rather than stopping when enough demonstrations were given to successfully train and validate the controller.

## Evaluating a trained controller

To rollout a trained controller:

`python rollout.py --domain dishes --task SetTable --model model --data eval`

Data from the evaluation experiment will be saved in the path provided to `--data`.
