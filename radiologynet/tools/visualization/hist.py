import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_data_hist(
    df: pd.DataFrame,
    axis=None,
    figsize: tuple = (5, 3),
    attribute="BodyPartExamined",
    sample_size=0,
    show_bar_labels: bool = True,
    sorted: bool = True
):
    """
    Retrieve a matplotlib.axes object ready to be displayed on a figure.
    On it, a histogram of all occurences of different values from df[attribute]

    Args:
            df - dataframe from which a histogram will be calculated
            axis - on which to draw the histogram.
                If None, will be instatiated from a new figure.
                Defaults to None.
            figsize - if axis is None, then this can be used to
                set the size of the generated figure. Defaults to
                (5, 3). Used only if axis is None.
            attribute - a key which is in `df`. Default value is
                'BodyPartExamined', which means
                that by default, a histogram of `df["BodyPartExamined"]`
                will be retrieved.
            sample_size - if less than or equal to 0, the entire df will be
                shown on histogram. If supplied with a value greater than 0,
                a random sample (of size `sample_size`) will be collected
                and displayed on the axes. Useful for testing purposes,
                especially if `df` is large.
            show_bar_labels (boolean, optional) - if this is True,
                then near each bar, the exact Y value will be written.
                Defaults to True.
            sorted (bool, optional) - if True, then the bars will be sorted
                by their Y value. Defaults to True.
    Returns:
            a matplotlib.axes object ready to be displayed on a figure.
    """
    sample = df.sample(sample_size) if sample_size > 0 else df
    # sort by attribute (not by instance count!)
    sample = sample.sort_values(attribute)
    sample = sample[attribute]
    if (axis is None):
        axis = plt.figure(figsize=figsize).gca()

    try:
        tmp = sample.value_counts(sort=sorted)
        axis.bar(tmp.index, tmp.values, color='skyblue', edgecolor="navy",)
        if show_bar_labels is True:
            fontsize = 10
            for x, y in zip(tmp.index, tmp.values):
                axis.text(
                    x, y, f'  {y}  ', rotation='vertical',
                    ha='center',
                    fontdict={'fontsize': fontsize}
                )
            # rescale the y axis so the text fits in
            axis.set_ylim([
                min(tmp.values.min(), 0),
                tmp.values.max() / len(str(tmp.values.max())) * fontsize / 1.5
            ])

        axis.tick_params(axis='x', labelrotation=90)
    except TypeError as e:
        # TODO try to figure out why one of the entries throws a TypeError
        print(e)

    axis.set_title('Histogram of %s' % (attribute))
    axis.tick_params(axis='x', labelrotation=90)
    return axis


def get_nonNaN_values_plot(
    df: pd.DataFrame,
    axis=None,
    figsize: tuple = (5, 3),
    in_percent=True,
    value_threshold=0,
    sample_size=0,
    sort: bool = False
):
    """
    Retrieve a matplolib.axes object ready to be displayed in a figure.
    On it, a bar plot of non-NaN values for all of the `df` dataframe
    attributes.

    Args:
            df - dataframe from which data will be extracted and counted
                in_percent - whether to calculate values as absolute
                counts of non-NaN values or to use percentages instead
                (the percentage of non-NaN values of each attribute)
            axis - on which to draw the plot.
                If None, will be instatiated from a new figure.
                Defaults to None.
            figsize - if axis is None, then this can be used to
                set the size of the generated figure. Defaults to
                (5, 3). Used only if axis is None.
            value_threshold - only values which are higher than or equal
                to this one will be included in the plot.
                If in_percent is True, please use percentages.
                If in_percent is False, please use absolute values.
                Default value is 0.
            sample_size - if less than or equal to 0, the entire df will be
                shown on histogram. If supplied with a value greater than 0,
                a random sample (of size `sample_size`) will be collected
                and displayed on the axes. Useful for testing purposes,
                especially if `df` is large.
    Returns:
            axis - a matplotlib.axes object ready to be displayed
                on a figure,
            non_nan - the resulting pandas.Series object of
                counted non-NaN values.
    """

    sample = df.sample(sample_size) if sample_size > 0 else df

    # first, count all of the non NaN values
    non_nan = sample.replace('', np.nan).count()

    # get percentage of non NaN values (if required)
    non_nan = non_nan / len(sample) if in_percent is True else non_nan

    # drop values which are lower than the threshold
    non_nan = non_nan[non_nan >= value_threshold]

    # finally, sort values if sorting was requested
    non_nan = non_nan.sort_values() if sort is True else non_nan

    if (axis is None):
        axis = plt.figure(figsize=figsize).gca()

    axis.bar(non_nan.index, non_nan.values,
             color='mediumseagreen', edgecolor='seagreen')
    axis.tick_params(axis='x', labelrotation=90)
    return axis, non_nan
