from dataclasses import dataclass
from typing import Tuple, Iterable
from collections import namedtuple
import tensorflow as tf
import matplotlib.pyplot as plt


_StatMeta = namedtuple(
    '_StatMeta',
    ['name', 'label', 'color'],
    defaults=['loss', None, None]
)


@dataclass
class _MlPlotMeta():
    stats: Iterable[_StatMeta]
    # only ylabel and ylim are availble for configuration;
    # xlim and xlabel are determined by epoch
    ylim: Tuple[float, float] = None
    ylabel: str = None
    title: str = None


def plot_ml_history_stats(
    history: tf.keras.callbacks.History,
    early_stopping: tf.keras.callbacks.EarlyStopping = None,
    plots: Iterable[_MlPlotMeta] = None,
):
    """Plot statistics of ML learning history.

    Args:
        history (tf.keras.callbacks.History): Learning history.

        early_stopping (tf.keras.callbacks.EarlyStopping, optional):
            If the Early stopping callback was used.
            If it was used and is provided, and the best epoch
            was not the last one,
            then the best epoch will be marked on the plot.
            Defaults to None.

        plots (Iterable[_MlPlotMeta], optional): Metadata of plots.
            If None, two plots will be plotted: Accuracy (with accuracy
            and val_accuracy)
            and Loss (with loss and val_loss)
            Defaults to None.

    Returns:
        Tuple[Figure,Axes]: the matplotlib Figure on which was plotted
        and the matplotlib Axes which contain the plots.
    """
    if (plots is None or len(plots) < 1):
        plots = [
            _MlPlotMeta(
                title='Training and Validation Loss',
                ylabel='Loss',
                stats=[
                    _StatMeta(name='loss', label='Training loss'),
                    _StatMeta(name='val_loss', label='Validation loss'),
                ]
            ),
            _MlPlotMeta(
                title='Training and Validation Accuracy',
                ylabel='Accuracy',
                stats=[
                    _StatMeta(name='accuracy', label='Training accuracy'),
                    _StatMeta(name='val_accuracy',
                              label='Validation accuracy'),
                ],
                ylim=[0, 1.1]
            )
        ]

    subplot_dim = (1, len(plots))
    figsize = (subplot_dim[1] * 10, subplot_dim[0] * 6)
    figure, ax = plt.subplots(
        subplot_dim[0], subplot_dim[1],
        figsize=figsize,
        constrained_layout=True
    )

    for i, plot in enumerate(plots):
        axis = ax[i]

        for stat_meta in plot.stats:
            stat_value = history.history[stat_meta.name]
            if(stat_meta.label is None):
                stat_meta.label = stat_meta.name
            axis.plot(history.epoch, stat_value,
                      c=stat_meta.color, label=stat_meta.label)

        axis.set_xlabel('Epoch')

        if(plot.title is not None):
            axis.set_title(plot.title)
        if(plot.ylim is not None):
            axis.set_ylim(plot.ylim)
        if(plot.ylabel is not None):
            axis.set_ylabel(plot.ylabel)

        if(early_stopping is not None and early_stopping.stopped_epoch > 0):
            axis.axvline(x=early_stopping.best_epoch,
                         c='grey', ls='-', label='Best Epoch')
        axis.legend()

    return figure, ax
