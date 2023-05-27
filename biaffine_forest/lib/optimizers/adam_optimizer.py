import tensorflow.compat.v1 as tf

from lib.optimizers.base_optimizer import BaseOptimizer

tf.disable_eager_execution()


#***************************************************************
class AdamOptimizer(BaseOptimizer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._adam = tf.train.AdamOptimizer(
        learning_rate=self.learning_rate,
        beta1=self.mu,
        beta2=self.nu,
        epsilon=self.epsilon)

  def minimize(self, loss, name=None):
    grads_and_vars = self._adam.compute_gradients(
        loss,
        gate_gradients=tf.train.Optimizer.GATE_OP,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        colocate_gradients_with_ops=True)

    if self.clip > 0:
      grads = [g for g, _ in grads_and_vars]
      grads, _ = tf.clip_by_global_norm(grads, self.clip)
      grads_and_vars = [(g, v) for g, (_, v) in zip(grads, grads_and_vars)]

    return self._adam.apply_gradients(
        grads_and_vars, global_step=self.global_step, name=name)
