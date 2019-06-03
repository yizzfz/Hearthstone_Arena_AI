from util import MovingAverage
from log import log


class BaseRunner:
    def __init__(self):
        self.avg_reward = MovingAverage(100)
        self.avg_loss = MovingAverage(100)

    def _train_info(self, epoch, reward):
        # moving averages
        text = [
            f'steps: {self.agent.step_cnt}',
            f'game epochs: {epoch}/{self.episodes}',
            f'train loss: {float(self.avg_loss):.5}',
            f'avg reward: {float(self.avg_reward):.5}',
            f'reward: {float(reward):.5}',
            f'epsilon: {self.agent.epsilon:.3}',
        ]
        log.info(', '.join(text), update=True)

    def _record(self):
        if self.agent.step_cnt % self.save_interval == 0:
            self.history.record({
                'steps': self.agent.step_cnt,
                'avg_reward': float(self.avg_reward),
                'loss': float(self.avg_loss),
            })
