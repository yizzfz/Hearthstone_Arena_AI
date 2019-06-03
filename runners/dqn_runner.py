from runners.base import BaseRunner
from log import log


class DQN_Runner(BaseRunner):
    def __init__(
            self, game, history, agent, memory,
            resume, episodes, batch_size, save_interval, gamma):
        super().__init__()
        self.game = game
        self.history = history
        self.agent = agent
        self.memory = memory

        self.episodes = episodes
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.gamma = gamma

        if resume is not None:
            self.agent.load(resume)

    def train(self):
        log.info(f'Training with {self.episodes}, starting ...')
        for i in range(self.episodes):
            state = self.game.reset()
            done = False
            loss = None
            while not done:
                state = self.game.state
                action = self.agent.select_action(state)

                transition, done = self.game.step(
                    int(action.to('cpu').numpy()))

                if len(self.memory) > self.batch_size:
                    batched = self.memory.sample(self.batch_size)
                    loss = self.agent.train(
                        batched, self.batch_size, self.gamma, i)
                    self.avg_loss.add(loss)
            reward = self.game.rewards
            self._train_info(i, reward)
            self._record()
            # agent.save_best(reward)
            self.agent.save()
            self.agent.scheduler.step()
            self.avg_reward.add(reward)
        self.game.env.close()

