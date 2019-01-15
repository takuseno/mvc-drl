class BaseController:
    def __init__(self,
                 metrics,
                 final_steps,
                 log_interval,
                 save_interval,
                 eval_interval):
        self.metrics = metrics
        self.final_steps = final_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval

    def step(self, obs, reward, done, info):
        raise NotImplementedError('implement step function')

    def should_update(self):
        raise NotImplementedError('implement should_update function')

    def update(self):
        raise NotImplementedError('implement update function')

    def stop_episode(self, obs, reward, info):
        raise NotImplementedError('implement update function')

    def should_log(self):
        return self.metrics.get('step') % self.log_interval == 0

    def log(self):
        raise NotImplementedError('implement log function')

    def should_save(self):
        return self.metrics.get('step') % self.save_interval == 0

    def save(self):
        self.metrics.save_model(self.metrics.get('step'))

    def is_finished(self):
        return self.metrics.get('step') >= self.final_steps

    def should_eval(self):
        return self.metrics.get('step') % self.eval_interval == 0
