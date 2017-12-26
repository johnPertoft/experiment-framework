import warnings

import tensorflow as tf


def early_stopping(metric, criteria, patience=1):
    """
    Return a function: current_metrics, all_previous_metrics -> bool
    which can be used to determine if the training loop should stop.
    :param metric: str, key for the metric to use to determine early stopping.
    :param criteria: fn: curr_val, prev_val -> bool indicating if performance has worsened.
    :param patience: tolerate degrading performance this many times.
    :return: early stopping callback function.
    """

    degrades = 0

    # TODO: Should be able to account for more than just the immediately prev value.
    # At least make it possible to extend. Also maybe make EarlyStopping into a class.
    def should_stop_training(curr_metrics, prev_metrics):
        nonlocal patience, degrades

        if not prev_metrics:
            return False

        if metric not in curr_metrics:
            warnings.warn(
                "Early stopping based on {} is not possible because it's not available. Try any of {}.".format(
                    metric, list(curr_metrics.keys())),
                RuntimeWarning)
            return False

        prev = prev_metrics[-1][metric]
        curr = curr_metrics[metric]

        has_worsened = criteria(curr, prev)
        if has_worsened:
            degrades += 1
        should_stop = has_worsened and degrades >= patience
        if should_stop:
            tf.logging.info("Early stopping based on: {}, {} vs {}".format(metric, curr, prev))
        return should_stop

    return should_stop_training
