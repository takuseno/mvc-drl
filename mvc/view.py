class View:
    def __init__(self, controller):
        self.controller = controller

    def step(self, obs, reward, done, info):
        if self.controller.should_update():
            self.controller.update()

        if self.controller.should_log():
            self.controller.log()

        if self.controller.should_save():
            self.controller.save()

        return self.controller.step(obs, reward, done, info)

    def stop_episode(self, obs, reward, info):
        self.controller.stop_episode(obs, reward, info)

    def is_finished(self):
        is_finished = self.controller.is_finished()
        if is_finished:
            self.controller.save()
        return is_finished

    def should_eval(self):
        return self.controller.should_eval()
