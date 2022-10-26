import typing
import tensorflow as tf
import numpy as np
from radiologynet.logging import log
import os
import torch
import datetime


def setup_tf_environment(
    gpus_idx: typing.List[int],
    verbose: bool = False
):
    """Setup GPUs and CPUs for training.

    Args:
        gpus_idx (List[int]), the indices of GPUs to use.
            If you want to work on CPU (and no GPU),
            then set this to an empty array.
        verbose (bool, optional): If True, logs useful
            logs. Defaults to False.
    """
    if len(gpus_idx) < 1:
        visible_devices = tf.config.get_visible_devices()
        log(f'Before hiding GPUs {visible_devices}', verbose=verbose)

        # Hide GPUs from visible devices
        tf.config.set_visible_devices([], 'GPU')
    else:
        log(
            f'Currently visible devices: ' +
            f'{tf.config.get_visible_devices()}', verbose=verbose)
        gpus = tf.config.list_physical_devices('GPU')
        gpus_to_use = [gpus[i] for i in gpus_idx]
        tf.config.set_visible_devices(gpus_to_use, 'GPU')
        for gpu_instance in gpus_to_use:
            tf.config.experimental.set_memory_growth(gpu_instance, True)
        # this helps with performance, but lowers down accuracy
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    visible_devices = tf.config.get_visible_devices()
    log(f'Using devices {visible_devices}', verbose=verbose)


def setup_torch_environment(
    gpus_idx: typing.List[int] = [],
    verbose: bool = False
):
    """Set up training environment for torch.

    Args:
        gpus_idx (typing.List[int]): Indices
            of GPUs to use. If empty, the CPU will
            be used instead. Defaults to [].
        verbose (bool, optional): Print useful logs.
            Defaults to False.

    Returns:
        _type_: _description_
    """
    if torch.cuda.is_available() and len(gpus_idx) >= 1:
        gpus_joined_as_str = ','.join(str(g) for g in gpus_idx)
        device = f'cuda:{gpus_joined_as_str}'
    else:
        device: str = 'cpu'
        # limit CPU usage
        torch.set_num_threads(5)

    log(f'Running on device:{device}', verbose=verbose)
    torch.device(device)
    return device


def to_tf_dataset(
    X, Y,
    shuffle_buffer_size=1000,
    batch_size=1000,
    save_to: str = None
) -> tf.data.Dataset:
    """Make a dataset using tf.Data API
    Args:
        X (Any): The input to predict fromt
        Y (Any): The true labels
        shuffle_buffer_size (int, optional): For shufflin
            the dataset. For optimal shuffling,
            set this to the length of the dataset. Defaults to 1000.
        batch_size (int, optional): Size of dataset batches.
            Defaults to 1000.
        save_to (str, optional): If provided, the datasets will be
            saved to this path and after then will be available
            for loading through `tf.data.experimental.load`. Defaults to None.

    Returns:
        tf.data.Dataset: The created dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    # dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size=batch_size,
                            num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if (save_to is not None):
        tf.data.experimental.save(dataset=dataset, path=save_to)
    return dataset


def load_tf_dataset(from_path: str) -> tf.data.Dataset:
    """Load a dataset which was previously saved using
    `tf.data.experimental.save`.

    Args:
        from_path (str): Path where to find the saved dataset.
            The path should lead to a folder.

    Returns:
        tf.data.Dataset: The loaded dataset.
    """
    dataset = tf.data.experimental.load(from_path)
    return dataset


def to_npy_dataset(
    X: typing.List,
    Y: typing.List,
    save_to: str = None
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Save X (source data) and Y(labels) to a numpy dataset.

    Args:
        X (Any): The source data.
        Y (Any): The labels.
        save_to (str, optional): Path to the dir where X and Y
            should be saved. If None, the datasets will not be saved.
            If set, then X will be saved to `[save_to]/X.npy` and
            Y will be saved `[save_to]/Y.npy`.
            Defaults to None.

    Returns:
        The numpy datasets as tuple, (X, Y)
    """
    X = np.array(X)
    Y = np.array(Y)

    if (save_to is not None):
        os.makedirs(save_to)
        X_save_path = os.path.join(save_to, f'X.npy')
        Y_save_path = os.path.join(save_to, f'Y.npy')

        np.save(X_save_path, arr=X, allow_pickle=True)
        np.save(Y_save_path, arr=Y, allow_pickle=True)

    return X, Y


def load_npy_dataset(from_path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Load a previously saved numpy dataset.

    Use this in conjuction with `to_npy_dataset`.

    Args:
        from_path (str): Path where to find the dataset

    Returns:
        X and Y (as Tuple), the source and the respective labels.
    """
    X_save_path = os.path.join(from_path, f'X.npy')
    Y_save_path = os.path.join(from_path, f'Y.npy')

    X = np.load(X_save_path, allow_pickle=True)
    Y = np.load(Y_save_path, allow_pickle=True)
    return X, Y


def save_torch_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int, path: str,
    best: bool = False,
    verbose: bool = False
):
    """Save torch model.

    Args:
        model (torch.nn.Module): the model to save.
        optimizer (torch.optim.Optimizer): optimizer
            to train the model.
        epoch (int): In how many epochs was this model trained.
        path (str): Path to where to save the model, without the
            extension. The extension will be added automatically.
            The extension is `.pth`.
        best (bool, optional): If True, `_best_model` will be
            added to the end of `path`, before the extension.
            Defaults to False.
        verbose (bool, optional): Print useful logs.
            Defaults to False.
    """
    _model = model
    # For paralell
    if isinstance(_model, torch.nn.DataParallel):
        _model = _model.module

    # Define saving state
    _state = {
        'time': str(datetime.datetime.now()),
        'model_state': _model.state_dict(),
        'model_name': type(_model).__name__,
        'optimizer_state': optimizer.state_dict(),
        'optimizer_name': type(optimizer).__name__,
        'epoch': epoch
    }

    # Save last model
    torch.save(_state, f'{path}.pth')
    log(f'Saving model to {path}', verbose=verbose)

    # Save best model
    if best is True:
        log(f'Saving best model to {path}!', verbose=verbose)
        torch.save(_state, path + '_best_model.pth')


def load_torch_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    best: bool = False
):
    """Load model state dictionary.

    Args:
        model (torch.nn.Module): model whose state dict
            should be loaded.
        optimizer (torch.optim.Optimizer): model optimizer.
        path (str): path to the state dict. It is expected
            that the state_dict loaded from this path
            is compatible with
            `radiologynet.learn.utils.save_torch_model()` function.
            The provided path will be concatenated with `.pth` extension.
        best (bool, optional): if True, loads the best model.
            Defaults to False.

    Returns:
        tuple: (model, optimizer) both with loaded states.
    """
    if best is True:
        path = f'{path}_best_model'
    path = f'{path}.pth'
    _state_dict = torch.load(path)
    model.load_state_dict(_state_dict['model_state'])
    optimizer.load_state_dict(_state_dict['optimizer_state'])

    return (model, optimizer)
