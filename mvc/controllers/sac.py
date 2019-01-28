from mvc.controllers.ddpg import DDPGController


class SACController(DDPGController):
    def _register_metrics(self):
        self.metrics.register('step', 'single')
        self.metrics.register('pi_loss', 'queue')
        self.metrics.register('v_loss', 'queue')
        self.metrics.register('q1_loss', 'queue')
        self.metrics.register('q2_loss', 'queue')
        self.metrics.register('reward', 'queue')

    def _record_update_metrics(self, *loss):
        self.metrics.add('v_loss', loss[0])
        self.metrics.add('q1_loss', loss[1][0])
        self.metrics.add('q2_loss', loss[1][1])
        self.metrics.add('pi_loss', loss[2])

    def log(self):
        step = self.metrics.get('step')
        self.metrics.log_metric('reward', step)
        self.metrics.log_metric('v_loss', step)
        self.metrics.log_metric('q1_loss', step)
        self.metrics.log_metric('q2_loss', step)
        self.metrics.log_metric('pi_loss', step)
