from SumoController import SumoController

class SumoEnvironment():
    def reset(self):
        self.sumo = SumoController()
        return self.sumo.getState()
    def step(self, action):
        cur_phase, state = self.sumo.getState()
        reward = self.sumo.makeAction(action)
        return cur_phase, state, reward, self.sumo.isFinised(), None
    def get_total_reward(self):
        return self.sumo.getTotalReward()
    def get_duration(self):
        return self.sumo.getDuration()
    def get_log_reward(self):
        return self.sumo.getLogReward()