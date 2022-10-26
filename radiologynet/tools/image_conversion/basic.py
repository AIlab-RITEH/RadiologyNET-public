from PIL import Image as PilImage
import os
from typing import Tuple
from radiologynet.tools.visualization.image import is_image_multidimensional
import numpy as np
from radiologynet.logging import log


def load_converted_image(
    disk_label: str,
    dcm_id: int,
    extension='png',
    return_2d: bool = False,
):
    """
    Load the DICOM image which was previously converted into PNG
     and return it as a list.

    Load the DICOM image which was previously converted into PNG
    according to the following rules:
        - DICOM images containing a single slice (one 2D image) are
            stored directly as-is, on the path which
            is of the same format as the one specified in get_
            path_from_id() function.
        - DICOM images with multiple slices (two or more 2D images)
            are stored within a folder
            whose name is the same as dcm_id. Inside the folder,
            each slice is its own separate image.
            An example path would be:
            `disk_label/dcm_id[0,-3]/dcm_id/slice_index.png`

    Args:
        disk_label (str) - location of the images
        dcm_id (int) - ID of the DICOM whose images have been
            converted according to the specified ruleset.
        return_2d (bool) - If True, returns the image of shape
            (img_width, img_height) and only takes the first slice of
            a multidimensional image. Defaults to False.

    Returns:
        The resulting image, in the shape of:
        `(n_slices, image_width, image_height)`
        or `(image_width, image_height)` if return_2d is True.
    """

    # this is the full path to the image (excluding the '.png' extension)
    # if the image contains a single slice
    # however if the image is multidimensional, then this is a path to a folder
    path_prefix = os.path.join(disk_label, str(dcm_id)[0:-3], str(dcm_id))
    result = [] if return_2d is False else None
    if os.path.isdir(path_prefix):
        # multidimensional image
        files = sorted(os.listdir(path_prefix))
        for slice_filename in files:
            slice_fullpath = os.path.join(path_prefix, slice_filename)
            img = PilImage.open(slice_fullpath)
            if(return_2d is True):
                result = img
                break
            result.append(img)
    else:
        # image with a single slice
        fullpath = '%s.%s' % (path_prefix, extension)
        img = PilImage.open(fullpath)
        if(return_2d is True):
            result = img
        else:
            result.append(PilImage.open(fullpath))

    return result


def save_image_slice(
    pixel_array: np.ndarray,
    fullpath: str,
    bits: int = 8,
    size: Tuple[int, int] = (None, None),
    verbose: bool = False,
):
    """Save a 2D image to disk.

    Args:
        pixel_array (np.ndarray): Pixel Array containing the image.

        fullpath (str): Full path to the new image (the full location
            where image should be saved). This should include the image
            name and extension.
        bits (int, optional): How many bits should the output image have.
            Defaults to 8.
        size (Tuple[int, int], optional): height and width of the
            new image, respectively.
            Defaults to (None, None).
        verbose (bool, optional): If True, prints out useful logs.
            Defaults to False.

    Raises:
        AssertionError: if the image provided is multidimensional.
    """
    image = pixel_array

    if(is_image_multidimensional(pixel_array_shape=pixel_array.shape)):
        raise AssertionError(
            'This function is used to save a single image slice.' +
            'The image must not be multidimensional.' +
            f'Found image of shape {pixel_array.shape}.'
        )

    if(bits == 16):
        image = image.astype(np.uint16)
    else:
        image = image.astype(np.uint8)
    image_to_save = PilImage.fromarray(image)

    if(size[0] is not None and size[1] is not None):
        curh, curw = pixel_array.shape
        aspect = curh / curw
        newh, neww = size
        if (curh > curw):
            neww = round(size[1] / aspect)
        else:
            newh = round(size[0] * aspect)

        # resize the core image
        image_to_save = image_to_save.resize((neww, newh), PilImage.BILINEAR)

        # once resized, now we have to (optionally) add zero-padding
        # first, calc the amount of padding which should be added
        # (padding is added by setting the top-left point where the image is
        # pasted onto an image of zeros)
        pad_left = int((size[1] - neww) // 2)
        pad_top = int((size[0] - newh) // 2)

        # now let's make a new image which consists of zeros only
        # (which means color is 0)
        result = PilImage.new(image_to_save.mode, (size[1], size[0]), color=0)
        # now we have an all-zero image. We should paste the dcm image
        # in the middle of it
        result.paste(image_to_save, (pad_left, pad_top))
        image_to_save = result

        log(
            f'Resizing image to {(image_to_save.height, image_to_save.width)}',
            verbose=verbose
        )

    log(f'Saving image to "{fullpath}"', verbose=verbose)
    image_to_save.save(fullpath)
    return


def save_image_as_png(
    dcm_id: int,
    pixel_array: np.ndarray,
    folder_where_to_save: str = './',
    verbose: bool = False,
    size: Tuple[int, int] = (None, None),
    exist_ok=False,
    bits: int = 8,
):
    """Save an entire image as a single or multiple PNGs.

    If image is  multidimensional, it will be saved as multiple PNGs.

    Args:
        dcm_id (int): ID of the DICOM image. Used for nomenclature and
            saving images named as their IDs.
        pixel_array (np.ndarray): The image itself.
        folder_where_to_save (str, optional): Where to save the images.
            Defaults to './'.
        verbose (bool, optional): Whether to print out useful logs.
            Defaults to False.
        size (Tuple[int, int], optional): Height and width of the image
            respectively; the original image will be resized to
            these dimensions.
            If not provided, the image will not be resized.
            Defaults to (None, None).
        exist_ok (bool, optional): If False and the exported image exists,
            a new image will not be created. If True, the existing image
            will be overriden. If the image does not exist, this is ignored.
            Defaults to False.
        bits (int, optional): How many bits should the output image have.
            Defaults to 8.
    """
    subfolder = str(dcm_id)[0:-3]

    if is_image_multidimensional(pixel_array_shape=pixel_array.shape):
        foldername = os.path.join(folder_where_to_save, subfolder, str(dcm_id))
        if(exist_ok is False and os.path.isdir(foldername)):
            log(f'Found {foldername}, skipping DCM ID: {dcm_id}',
                verbose=verbose)
        else:
            # create a folder which will contain all of the image's slices
            os.makedirs(foldername, exist_ok=True)

            for index_of_slice, image_slice in enumerate(pixel_array):

                # save all of the individual slices as PNG in the image folder
                fullpath = os.path.join(
                    foldername, 's%04d.png' % (index_of_slice))

                save_image_slice(
                    image_slice,
                    fullpath,
                    bits=bits,
                    verbose=verbose,
                    size=size,
                )
    else:
        fullpath = os.path.join(folder_where_to_save,
                                subfolder, f'{dcm_id}.png')
        if(exist_ok is False and os.path.isfile(fullpath)):
            log(f'Found {fullpath}, skipping DCM ID: {dcm_id}',
                verbose=verbose)
        else:
            os.makedirs(os.path.join(
                folder_where_to_save, subfolder), exist_ok=True)
            save_image_slice(
                pixel_array,
                fullpath,
                bits=bits,
                verbose=verbose,
                size=size,
            )
