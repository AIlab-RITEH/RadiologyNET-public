from typing import Iterable, Union
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import math

from radiologynet.tools.image_conversion.window import window_dcm_image


def is_image_multidimensional(
    dcm_obj: pydicom.Dataset = None,
    pixel_array_shape: Iterable = None
):
    """
    Check if the DICOM image is multidimensional.
    For example, ultrasounds, CT and MRI images can be threedimensional.
    A CT image can be of shape (2, 512, 512), which means it consists
    of two greyscale images.
    Parameters:
        dcm_obj - DICOM object whose image needs to be checked.
            If not specified, then pixel_array_shape must be provided.
            Defaults to None.
        pixel_array_shape - And array of elements which describe the
            shape of an array.
            Defaults to None.
    Returns:
        True if the image consists of more than two dimensions.
            Otherwise False.
    Throws:
        Assertion error - if neither dcm_obj or pixel_array shape
            were provided.
    """
    shape = None
    if dcm_obj is not None:
        shape = dcm_obj.pixel_array.shape
    else:
        shape = pixel_array_shape

    if shape is None:
        raise AssertionError(
            'Either dcm_obj or pixel_array_shape must be provided!')

    # dcm_obj.Rows and dcm_obj.Columns contain the image size.
    # For example, if CT image has the shape (2, 512, 512)
    # then Rows=512 and Columns=512.
    result = True
    if len(shape) < 3:
        # image has less than 2 dimensions
        # so it's surely not multidimensional
        result = False
    elif shape[2] <= 3:
        # if the third dimension has less than three channels
        return False
    return result


def is_image_rgb(
    dcm_obj: pydicom.Dataset = None,
    pixel_array_shape: Iterable = None
):
    shape = None
    if dcm_obj is not None:
        shape = dcm_obj.pixel_array.shape
    else:
        shape = pixel_array_shape

    if shape is None:
        raise AssertionError(
            'Either dcm_obj or pixel_array_shape must be provided!')

    result = False
    if(shape[-1] == 3):
        result = True
    return result


def plot_dicom_image(
    dcm: pydicom.Dataset,
    return_pixel_array: bool = None,
    layer_index: Union[str, int] = 'auto',
    do_windowing: bool = True,
    verbose: bool = False,
):
    """
    Plot DICOM image and output (print) its basic properties.
    Parameters:
        dcm - DICOM object which should be shown.
        return_pixel_array - if True, this function returns the DICOM object's
            pixel_array values and does not create a maptlotlib plot.
            "None" by default.
        layer_index - if the image is multidimensional (such as ultrasounds),
            than this is the index of the image's layer which should be shown.
            For example, an image can have the dimensions: (76, 512, 512, 3)
            and by specifying the layer_index as `layer_index=30`,
            then the image at
            dicom_object.pixel_array[30] will be shown, which
            has the shape (512, 512, 3).
            * If `'auto'`, then if image is multidimensional, a random
            layer_index will be chosen automatically.
            * If `None`, the image dimensionality
            is ignored and the image is treated as 2D.
            Defaults to `'auto'`.
        do_windowing (bool, optional) - should image windowing using
            WindowCenter and WindowWidth be performed.
            See radiologynet.tools.image_conversion.window for details.
            Defaults to True.
        verbose - if True, will print out potentially useful info.
    Returns:
        optionally, if return_pixel_array is set to True, the DICOM
        object's pixel_array will be returned.
    """

    rows = int(dcm.Rows)
    cols = int(dcm.Columns)

    if verbose is True:
        print('{:>15}: {rows:d} x {cols:d}, {size:d} bytes'.format(
            'Image size',
            rows=rows,
            cols=cols,
            size=len(dcm.PixelData)
        ))

        print('{:>15}: {shape}'.format(
            'Image shape',
            shape=dcm.pixel_array.shape
        ))

        if 'PixelSpacing' in dcm:
            print('{:>15}: {}'.format(
                'Pixel spacing',
                dcm.PixelSpacing
            ))

        if(dcm.get('TransferSyntaxUID') is None):
            default_transsyntax = pydicom.uid.ImplicitVRLittleEndian
            dcm.file_meta.TransferSyntaxUID = default_transsyntax
            print('TransferSyntaxUID Missing :: ',
                  'Set Default Transfer Syntax to Implicit VR Little Endian')

        print('{:>15}: {}'.format(
            'Slice location',
            dcm.get('SliceLocation', '(missing)')
        ))

    if layer_index is 'auto':
        if is_image_multidimensional(dcm_obj=dcm):
            num_layers = dcm.pixel_array.shape[0]
            layer_index = num_layers // 2
        else:
            layer_index = None

    if layer_index is not None:
        if(verbose is True):
            print('{:>15}: {}'.format(
                'Layer index',
                layer_index
            ))

    # if windowing should be performed then window the image
    if do_windowing is True:
        pixel_array = window_dcm_image(
            dcm_obj=dcm,
            verbose=verbose
        )
    else:
        # if there is no windowing then use raw data
        pixel_array = dcm.pixel_array

    # if image is multidimensional then get a single layer of it
    pixel_array = pixel_array if \
        layer_index is None else pixel_array[layer_index]

    if return_pixel_array:
        return pixel_array

    plt.imshow(
        pixel_array,
        cmap='bone'
    )
    plt.show()


def plot_mosaic(
    dcms: Union[Iterable[pydicom.Dataset], Iterable[int]],
    cnt: int = 9,
    cmap: str = 'bone',
    get_title=None,
    figsize: tuple = None,
    return_figure: bool = False,
    path_to_images: str = None,
    do_windowing: bool = True,
):
    """Plot a mosaic of DCM images.

    Args:
        dcms (Iterable[pydicom.Dataset], Iterable[int]): DCMs to plot.
            An array containing either DICOM objects of DICOM IDs. If
            it contains DCM IDs, then path_to_images must be provided.
        cnt (int, optional): How many images to show in mosaic.
            Defaults to 9.
        cmap (str, optional): colormap to use in matplotlib subplots.
            Defaults to 'bone'.
        get_title (function, optional): a callback function to format
            each subplot title. If None, then it will just tell you the
            image Modality.
            Arguments passed to `get_title`:
                1) i - index of the current DCM,
                2) dcm - the DCM object which is being plotted on the subplot.
            Defaults to None.
        figsize (tuple, optional): Figure size, a tuple of two elements
            (width, height). If None, it's calculate automatically.
            Defaults to None.
        return_figure (bool, optional): If True, `(figure, axes)` is returned
            and `plt.show()` is not called.
            Defaults to False.
        path_to_images(str, optional): If DICOM IDs were passed instead of
            DICOM objects, then this has to contain the path to the folder
            where the DICOMs are located (according to the rules specified in
            radiologynet.tools.raw.io).
        do_windowing (bool, optional) - should windowing be perfomed.
            Defaults to True.

    Throws:
        AssertionError: If DICOM IDs were passed, but path_to_images wasn't.

    Returns:
        figure: plt.figure, if `return_figure` is True.
        axes: plt.Axes, if `return_figure` is True.
    """
    is_array_of_ids = False
    try:
        _ = dcms[0].get('Modality')
        is_array_of_ids = False
    except Exception as e:
        is_array_of_ids = True

    assert (is_array_of_ids is True and path_to_images is None) is False,\
        'Path to the images must be provided if DCM IDs were passed!'

    import radiologynet.tools.raw.io as raw_io

    if is_array_of_ids is True:
        dcm_ids = dcms
        # if there are more dicom ids passed than there should be shown on
        # mosaic, then remove any excess dicoms
        dcm_ids = dcm_ids if len(dcm_ids) <= cnt else dcm_ids[:cnt]

        dcms = [
            raw_io.get_dicom_by_id(foldername=path_to_images, dcm_id=dcm_id)
            for dcm_id in dcm_ids
        ]

    if(get_title is None):
        def get_title(i, dcm): return f'Modality: {dcms[i].get("Modality")}'

    def show_img(k, dcm, axes: plt.Axes):
        try:
            pixel_array = plot_dicom_image(
                dcm, return_pixel_array=True, do_windowing=do_windowing)
        except AssertionError as e:
            return
        axes.imshow(pixel_array, cmap=cmap)
        axes.set_title(get_title(k, dcm))

    cnt = np.min([cnt, len(dcms)])
    height = math.ceil(math.sqrt(cnt))
    width = math.ceil(cnt / height)

    figsize = figsize if figsize is not None else (width * 2.5, height * 2.5)

    fig, axes = plt.subplots(
        height,
        width,
        constrained_layout=True,
        figsize=figsize
    )

    # in case there is only 1 image to plot
    # then axes isn't an array of AxesSubplot.
    # So instead of being a single instance of AxesSubplot
    # convert into ndarray[AxesSubplot] so the rest of the code works.
    # Note that this array will have only one element inside.
    axes_iterable = axes.flat if \
        isinstance(axes, np.ndarray) else np.array([axes])

    for i in range(height):
        for j in range(width):
            k = i * width + j
            if(k >= cnt):
                break
            dcm = dcms[k]
            show_img(k, dcm, axes_iterable[k])

    if return_figure is True:
        return fig, axes

    plt.show()
