class View:
    def __init__(self, controller):
        self.controller = controller

    def step(self, obs, reward, done):
        if self.controller.should_update():
            self.controller.update()
        return self.controller.step(obs, reward, done)
