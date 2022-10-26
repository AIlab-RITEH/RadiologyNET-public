import pydicom
import numpy as np
from radiologynet.logging import log


def transform_to_hu(
    dcm_obj: pydicom.Dataset
) -> np.ndarray:
    """
    If intercept and slope exists, the image pixels'
    will be transformed to appropriate values.
    If interecept and slope are lists, only first value is taken.

    Args:
        dcm_obj (pydicom.Dataset) - DICOM Object

    Returns:
        the transformed pixel_array
    """
    _intercept = dcm_obj.RescaleIntercept
    _slope = dcm_obj.RescaleSlope
    if isinstance(_slope, list):
        _slope = _slope[0]
    if isinstance(_intercept, list):
        _intercept = _intercept[0]

    image = dcm_obj.pixel_array
    image = image * _slope + _intercept
    return image


def window_dcm_image(
    dcm_obj: pydicom.Dataset,
    bits: int = 8,
    verbose: bool = False
):
    """Perform DICOM image windowing.

    DICOM Images are windowed using WindowCenter and
    WindowWidth attributes. All pixels which do not fit in the
    window value are set to 0 (if lower than bottom limit of window)
    or to max_pixel_value (if higher than top limit of window).
    max_pixel_value is calculated from the 'bits' attribute as
    `2 ^{bits} - 1`.

    If there are multiple WindowCenter or WindowWidth values (i.e.
    these attributes as arrays), then the first value is used
    for windowing.

    Args:
        dcm_obj (pydicom.Dataset): The DCM object.
        bits (int, optional): How many bits should the output image have.
            This determines the possible highest pixel value.
            Defaults to 8 (which means 255 is the highest pixel value).
        verbose (bool, optional): If True, prints useful logs.
            Defaults to False.

    Returns:
        ndarray: the transformed image.
    """
    center, width = None, None
    try:
        width = dcm_obj.WindowWidth
        center = dcm_obj.WindowCenter
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        img = dcm_obj.pixel_array
    except Exception as e:
        log('Missing one of the following:' +
            ' WindowCenter, WindowWidth, PixelData', verbose=verbose)

    assert (
        center is not None
        and width is not None
        and img.shape[0] >= 1
    ), 'WindowCenter, WindowWidth or PixelData not valid'

    max_pixel_intensity = 2 ** bits - 1
    # apply slope and intercept if available
    if (dcm_obj.get('RescaleIntercept') and dcm_obj.get('RescaleSlope')):
        img = transform_to_hu(dcm_obj=dcm_obj)

    # perform image windowing
    retval = np.piecewise(
        img,
        [
            # conditions to check
            img <= (center - (width / 2)),
            img > (center + (width / 2))
        ],
        [  # what to apply on each condition
            0,  # where first condition true
            max_pixel_intensity,  # where second condition true
            lambda img: (  # where none of the conditions are true
                (img - center + width / 2) / width * max_pixel_intensity
            )
        ]
    )

    return retval
