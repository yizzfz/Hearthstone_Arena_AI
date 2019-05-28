import sys
import click
import torch

from environment import Game
from agents import agent_factory
from util import ReplayMemory

environments_to_names = {
    'cartpole': 'CartPole-v0',
    'mountaincar': 'MountainCar-v0',
    'space_invader': 'SpaceInvaders-v0', # an atari game
}


@click.group()
@click.version_option()
def cli():
    """
        Witch --- An automatic Witch that plays hearthstone Arena.
        Authors:
            Aaron Zhao, yaz21@cam.ac.uk
            Han Cui,
    """


@click.command(help="To train the agent")
@click.argument('method')
@click.argument('environment')
@click.option('resume', '--resume', default=None, type=str)
@click.option('episodes', '--episodes', default=None, type=int)
@click.option('lr', '--lr', default=0.1, type=float)
@click.option('lr_episodes', '--lr_episodes', default=50, type=int)
@click.option('min_lr', '--min_lr', default=0.00001, type=float)
@click.option('eval_only', '--eval_only', is_flag=True)
@click.option('replay_width', '--replay_width', default=1000, type=int)
@click.option('num_episodes', '--num_episodes', default=100, type=int)
def train(
        method, environment,
        resume, episodes,
        lr, lr_episodes, min_lr,
        eval_only, replay_width, num_episodes):
    memory = ReplayMemory(replay_width)
    game = Game(
        name=environments_to_names[environment],
        memory=memory, render=False)
    init_state, state_shape = game.get_state(True)
    n_actions = game.env.action_space.n
    agent_cls = agent_factory[method]
    agent = agent_cls(state_shape, n_actions)

    for i in range(num_episodes):
        pass

    game.env.close()


@click.command(help="You can play with the agent")
@click.argument('environment')
def play(environment):
    game = Game(name=environments_to_names[environment], render=True)
    done = False
    try:
        while not done:
            action = click.prompt('Please enter an action (0, 1, 2, 3..)')
            done = game.step(int(action))
        print('[INFO] done ...')
    except KeyboardInterrupt:
        print("[INFO] quiting ...")
        exit()


cli.add_command(train)
cli.add_command(play)


if __name__ == '__main__':
    cli()
