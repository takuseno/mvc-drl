from mvc.controllers.ddpg import DDPGController


class TD3Controller(DDPGController):
    def _record_update_metrics(self, *loss):
        self.metrics.add('critic_loss', loss[0])
        if loss[1] is not None:
            self.metrics.add('actor_loss', loss[1])

    def update(self):
        assert self.should_update()

        # sample batch from replay buffer
        batch = self.buffer.fetch(self.batch_size)

        # delayed policy update
        update_actor = self.metrics.get('step') % 2 == 0

        # update
        loss = self.network.update(**batch, update_actor=update_actor)

        # record metrics
        self._record_update_metrics(*loss)

        return loss
