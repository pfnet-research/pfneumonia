from chainer.training import extension
from chainer.training.triggers.interval_trigger import IntervalTrigger


def get_total_progress(trainer):
    """Returns a number in [0, 1] which describes how much of the total training is finished."""

    stop_trigger = trainer.stop_trigger
    assert isinstance(stop_trigger, IntervalTrigger)

    if stop_trigger.unit == 'epoch':
        return trainer.updater.epoch_detail / stop_trigger.period
    elif stop_trigger.unit == 'iteration':
        return trainer.updater.iteration / stop_trigger.period
    else:
        assert False


class LearningRateScheduler(extension.Extension):

    # Based on ExponentialShift
    def __init__(self, func, attr='lr', target=None, optimizer=None):
        self._attr = attr
        self._func = func
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def __call__(self, trainer):
        optimizer = self._get_optimizer(trainer)
        updater = trainer.updater
        val = self._func(updater.epoch, updater.epoch_detail, updater.iteration,
                         get_total_progress(trainer))
        self._update_value(optimizer, val)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
