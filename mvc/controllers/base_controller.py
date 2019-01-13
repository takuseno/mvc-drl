class BaseController:
    def step(self, obs, reward, done):
        raise NotImplementedError('implement step function')

    def should_update(self):
        raise NotImplementedError('implement should_update function')

    def update(self):
        raise NotImplementedError('implement update function')

    def stop_episode(self, obs, reward):
        raise NotImplementedError('implement update function')

    def should_log(self):
        raise NotImplementedError('implement should_log function')

    def log(self):
        raise NotImplementedError('implement log function')

    def is_finished(self):
        raise NotImplementedError('implement is_finished function')
