class TrainView:
    def __init__(self, controller):
        self.controller = controller

    def step(self, obs, reward, done):
        if self.controller.should_update():
            self.controller.update()
        return self.controller.step(obs, reward, done)

    def stop_episode(self, obs, reward):
        self.controller.stop_episode(obs, reward)


class EvalView:
    def __init__(self, controller):
        self.controller = controller

    def step(self, obs, reward, done):
        return self.controller.step(obs, reward, done)

    def stop_episode(self, obs, reward):
        self.controller.stop_episode(obs, reward)
