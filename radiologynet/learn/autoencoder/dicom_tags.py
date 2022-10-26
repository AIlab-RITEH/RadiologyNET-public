import pandas as pd
from radiologynet.tools.feature_extractors.dicom_tags \
    import DicomTagFeatureExtractor
from radiologynet.logging import log
import numpy as np
import typing
# import tensorflow as tf
import torch
import torch.nn as nn
import os
import shutil
import matplotlib.pyplot as plt
import csv
import radiologynet.learn.utils as utils


class LayerConfig:
    size: int
    dropout: float
    type: str
    kernel_size: int
    nfilters: int

    def __init__(
        self,
        size: int = 0,
        dropout: float = 0.0,
        type: str = 'Dense',
        kernel_size: int = 0,
        nfilters: int = 0
    ) -> None:
        self.size = size
        self.dropout = dropout
        self.type = type
        self.kernel_size = kernel_size
        self.nfilters = nfilters

    def __str__(self) -> str:
        return (f'LayerConfig(size={self.size}, ' +
                f'dropout={self.dropout}, ' +
                f'type="{self.type}", ' +
                f'kernel_size={self.kernel_size}, ' +
                f'nfilters={self.nfilters})'
                )


class AEModel(nn.Module):
    train_losses: typing.List[float] = []
    """Losses gathered during training.
    Empty if training has not been performed.
    """
    val_losses: typing.List[float] = []
    """Losses gathered during validation in each training epoch.
    Empty if training has not been performed.
    """
    _bottleneck_size: int = None
    """ Size of the layer inbetween the encoder and the decoder.
    Size of the latent representation. Output size
    of the bottleneck layer.
    """
    _bottleneck_input: int = None
    """Input size of the bottleneck layer."""
    _loss_func = nn.MSELoss()
    """Loss function used for training the model."""

    encoder: nn.Sequential
    decoder: nn.Sequential
    _encoder_layers: typing.List[nn.Module]
    _decoder_layers: typing.List[nn.Module]

    _fc1: nn.Linear
    _fc_bottle: nn.Linear
    _fc2: nn.Linear

    _layer_config: typing.List[LayerConfig]
    """Configuration of the model."""

    _n_epochs: int = 10
    """How many epochs were used to train this model."""

    def __init__(
            self,
            layer_config: typing.List[LayerConfig],
            in_size: int,
            out_size: int,
            bottleneck_size: int = None
    ):
        super().__init__()
        self._layer_config = layer_config

        def get_layers(
            _layer_config: typing.List[LayerConfig],
            input_size: int
        ):
            _layers = []
            for lyr in _layer_config:
                _layers.append(nn.Linear(
                    in_features=input_size,
                    out_features=lyr.size
                ))
                _layers.append(nn.LeakyReLU(inplace=True))
                _layers.append(nn.BatchNorm1d(lyr.size))
                if lyr.dropout > 0:
                    _layers.append(nn.Dropout(lyr.dropout))
                # update the input size of the following layer
                input_size = lyr.size
            return _layers

        # encoder part of the network
        _encoder_layers = get_layers(layer_config, input_size=in_size)
        # bottleneck_input --> how many features are
        # inputted into the bottleneck layer.
        self._bottleneck_input = layer_config[-1].size
        self._bottleneck_size = bottleneck_size

        # the bottleneck which will serve for getting the
        # latent representation
        self._fc1 = nn.Linear(self._bottleneck_input, self._bottleneck_size)
        # THIS is the bottleneck layer
        self._fc_bottle = nn.Linear(
            self._bottleneck_size, self._bottleneck_size)
        # layer for passing data back to conv1d
        self._fc2 = nn.Linear(self._bottleneck_size, self._bottleneck_input)

        # decoder part of network
        _reversed_lyrs = list(reversed(layer_config))
        _decoder_layers = get_layers(
            _reversed_lyrs[1:], input_size=_reversed_lyrs[0].size)
        _decoder_layers.append(
            nn.Linear(_reversed_lyrs[-1].size, out_features=out_size)
        )

        self.encoder = nn.Sequential(*_encoder_layers)
        self.decoder = nn.Sequential(*_decoder_layers)
        self._encoder_layers = _encoder_layers
        self._decoder_layers = _decoder_layers

    def get_latent(self, x):
        """Get the latent t representaion of x.

        Args:
            x: the data which should be transformed
                into a latent representation.

        Returns:
            tuple: The first element is the latent represenation z.
                The second element is another tuple describing the
                original shape of x.
        """
        x = self.encoder(x)

        batch, n_features = x.shape
        hidden = self._fc1(x)

        z = self._fc_bottle(hidden)  # now we have the latent representation

        # return the latent rep and the original shape
        # we need the original shape to reshape the data
        # for decoder input
        return z, (batch, n_features)

    def forward(self, x):
        z, (batch, n_features) = self.get_latent(x)

        x = self._fc2(z)

        x = self.decoder(x)
        reconstruction = torch.sigmoid(x)
        return reconstruction


class DicomTagAutoEncoder():
    data: pd.DataFrame
    """The data which will be used for training."""
    workdir_path: str
    """The directory where all files and models will be saved."""
    model: AEModel
    """The NN model."""
    _optimizer: torch.optim.Optimizer
    """Optimizer used for training."""

    def __init__(self, workdir_path: str, data: pd.DataFrame = None) -> None:
        """Init DCM tag AE.

        Args:
            workdir_path (str): path to where the datasets and
                trained models will be saved.
            data (pd.DataFrame, optional): The data. Defaults to None.
        """
        self.data = data
        self.workdir_path = workdir_path

    def preprocess_data(
        self,
        attribute_to_balance='Modality',
        nsamples_per_attribute=None,
        scaler_type: str = 'MinMax',
        encoder_type: str = 'Label',
        random_state: int = 1,
        verbose: bool = False,
        unique_values_threshold: int = 2,
        cardinality_threshold: int = 50,
        percent_filled_threshold: float = 0.2,
        impute_missing: bool = True
    ):
        """Perform data balancing, scaling, encoding... The operations
        are done inplace.

        Args:
            attribute_to_balance (str, optional): Attribute which
                should be balanced throughout the dataset.
                Defaults to 'Modality'. If None, no rebalancing will occur.

            nsamples_per_attribute (int, optional): How many samples of
                each value the `attribute_to_balance` column should have.
                Defaults to None.

            scaler_type (str, optional): Which scaler to use.
                If not set, no scaling will be performed.
                Defaults to 'MinMax'.

            encoder_type (str, optional): Which encoder to use.
                For more info, check out
                `radiologynet.tools.feature_extactors.dicom_tags.encode`.
                If not set, no encoding will be performed.
                Defaults to 'Label'.

            verbose (bool, optional): If true, prints useful logs.
                Defaults to False.

            random_state (int, optional): For reproducibility of
                random events. Defaults to 1.

            unique_values_threshold (int, optional) - if set,
                columns with less than this number of distinct values
                will be dropped. Defaults to 2, meaning that each column
                will have at least 2 distinct values.

            cardinality_threshold (int, optional) - if set,
                each categorical variable with more than this amount
                of distinct values will be dropped.
                Defaults to 50, meaning than each categorical variable
                with cardinality higher than 50 will be dropped.
        """
        features = DicomTagFeatureExtractor(self.data.copy())
        if (attribute_to_balance is not None):
            features.resample_to_have_equal_of(
                attribute=attribute_to_balance,
                inplace=True,
                nsamples=nsamples_per_attribute,
                random_state=random_state
            )

        # fix body part examined
        log(f'Parsing BodyPartExamined...', verbose=verbose)
        features.fix_BodyPartExamined(inplace=True, drop_others=False)

        if (attribute_to_balance is not None):
            # since we dropped some rows, we need to rebalance the data
            features.resample_to_have_equal_of(
                inplace=True,
                attribute=attribute_to_balance,
                random_state=random_state
            )

        if (unique_values_threshold is not None):
            features.drop_columns_with_insufficient_unique_values(
                inplace=True,
                threshold=unique_values_threshold,
                verbose=verbose
            )

        log('Parsing arraylike and numeric values...', verbose=verbose)
        features.parse_arraylike_and_numeric_values_from_columns(inplace=True)
        # get different statistics related to columns
        # this will be useful when imputing data, dropping columns etc.
        if (cardinality_threshold is not None
                or percent_filled_threshold is not None):
            log('Checking cardinality and/or fill rate...', verbose=verbose)
            stats = features.get_stats_of_columns()
            cols_with_high_card: typing.List[str] = []
            cols_with_low_fillrate: typing.List[str] = []
            for col in stats.index:
                stats_for_col = stats.loc[col]
                if (cardinality_threshold is not None and
                        stats_for_col.DataType == 'str' and
                        stats_for_col.UniqueValues > cardinality_threshold):
                    # this is above the cardinality threshold
                    # so drop the column entirely
                    cols_with_high_card.append(col)
                    # if this value is already marked for dropping
                    # then there is no need to check it further
                    # so continue onto the next column
                    continue
                if (
                    percent_filled_threshold is not None and
                    stats_for_col.PercentFilled < percent_filled_threshold
                ):
                    cols_with_low_fillrate.append(col)
            if cardinality_threshold is not None:
                log(
                    f'Dropping {len(cols_with_high_card)} columns because ' +
                    f'they have cardinality higher than ' +
                    f'{cardinality_threshold}.',
                    verbose=verbose
                )
            if percent_filled_threshold is not None:
                log(
                    f'Dropping {len(cols_with_low_fillrate)} columns due' +
                    f' to having ' +
                    f'a non-empty value in less than ' +
                    f'{percent_filled_threshold*100}% of data.',
                    verbose=verbose
                )

            cols_to_drop = [*cols_with_high_card, *cols_with_low_fillrate]
            features.tags.drop(cols_to_drop, axis=1, inplace=True)

        if encoder_type is not None:
            # refresh features.tags statistics
            # because after parsing arraylike values
            # the columns might have different names.
            stats = features.get_stats_of_columns()
            # also, encode the features AFTER
            # the stats have been caluclated.
            features.encode(encoder_type, inplace=True, verbose=False)
            if impute_missing is True:
                cat_varnames = stats.query(
                    'DataType=="str"').sort_index().index
                features.impute_missForest(
                    inplace=True, verbose=verbose, cat_varnames=cat_varnames)
        if scaler_type is not None:
            features.scale(type=scaler_type, inplace=True)

        self.data = features.tags
        log(f'All done! The final data shape is {np.shape(self.data)}',
            verbose=verbose)

    def to_dataset(
        self,
        train_ids: typing.List[int],
        test_ids: typing.List[int],
        val_ids: typing.List[int],
        save_to: str = None,
        random_state: int = 1,
        verbose: bool = False
    ):
        """Save the train, test and validation datasets
        as npy datasets.

        Args:
            save_to (str): Where to save the dataset.
                Training dataset will be saved to {save_to}/data/train,
                Validation dataset will be saved to {save_to}/data/val,
                Test dataset will be saved to {save_to}/data/test.
                If None, then self.workdir_path will be used.
                Defaults to None.
            train_ids (List[int]): list of IDs which are in the
                train subset.
            test_ids (List[int]): list of IDs which are in the
                test subset.
            val_ids (List[int]): list of IDs which are in the
                val subset.
            random_state (float, optional): for RNG. Defaults to 1.0.
            verbose (bool, optional): If true, prints useful logs.
                Defaults to False.
        """
        if (save_to is None):
            save_to = self.workdir_path

        dataset_meta = zip([train_ids, test_ids, val_ids],
                           ['train', 'test', 'val'])

        for dataset_ids, ds_name in dataset_meta:
            path = os.path.join(save_to, 'data', ds_name)
            os.makedirs(path, exist_ok=True)

            dataset = self.data.loc[dataset_ids]

            # if this path already exists, delete it
            # because who knows what's in there
            if os.path.exists(path):
                log(f'Found something at {path}. Deleting it...',
                    verbose=verbose)
                shutil.rmtree(path)
            # save all of the IDs to CSVs for reproduction
            csv_path = os.path.join(save_to, 'data', f'{ds_name}-IDs.csv')
            f = open(csv_path, 'w')
            writer = csv.writer(f)
            writer.writerow(dataset_ids)
            f.close()
            log(
                f'Saved {ds_name} IDs to {csv_path}.',
                verbose=verbose
            )

            utils.to_npy_dataset(
                X=dataset,
                Y=dataset,
                save_to=path
            )

            log(f'Saved dataset "{ds_name}" to {path}', verbose=verbose)

    def fit_epoch(
        self,
        model: AEModel,
        optimizer: torch.optim.Optimizer,
        xdataloader: torch.utils.data.DataLoader,
        ydataloader: torch.utils.data.DataLoader,
        device: torch.device,
        validation: bool = False,
    ):
        """Fit the model to the data -- perform a single
        epoch of training.

        Args:
            model (torch.nn.Module): the model to fit to.
            xdataloader (torch.utils.data.DataLoader):
                input values to train on.
            ydataloader (torch.utils.data.DataLoader):
                desired output values of the model.
            validation (bool, optional): Should be True if this fitting
                performed on validation and not training data.
                If False, loss will be propagated backwards and
                model parameteres will be updated.
                Defaults to False.

        Returns:
            float: the calculated total loss of the epoch.
        """
        total_epoch_loss: float = 0.0
        for x, y in zip(xdataloader, ydataloader):
            x = x.to(device)
            y = y.to(device)
            # this for loop goes over all the batches
            if validation is False:
                # begin training of batch: set all gradients to 0.
                optimizer.zero_grad()
            reconstruction = model(x)
            loss = model._loss_func(reconstruction, y)

            total_epoch_loss += loss

            if validation is False:
                loss.backward()
                optimizer.step()
        # the total epoch loss is the average of batch losses
        # so divide the sum of losses by the number of batches
        total_epoch_loss /= len(xdataloader.dataset)
        return total_epoch_loss

    def train(
        self,
        layer_configs: typing.List[LayerConfig],
        bottleneck_size: int = None,
        learning_rate: float = 0.01,
        batch_size=32,
        epochs=200,
        return_compiled_model: bool = False,
        verbose: bool = False,
        model_path: str = None,
        gpu_idx: typing.List[int] = [],
        early_stopping: int = None,
    ):
        """Perform a full training cycle.

        Args:
            layer_configs (typing.List[LayerConfig]): Configuration
                of the model.
            bottleneck_size (int, optional): The size
                of the bottleneck layer. If None, then the
                input size will be halved and used as bottleneck
                size.
                Defaults to None.
            learning_rate (float, optional): The learning rate.
                Defaults to 0.01.
            batch_size (int, optional): Batch size. Defaults to 32.
            epochs (int, optional): Number of epochs to train for.
                Defaults to 200.
            return_compiled_model (bool, optional): If True, will return
                only the built model (not trained).
                Defaults to False.
            verbose (bool, optional): Print useufl logs. Defaults to False.
            model_path (str, optional): Path to where to save the model.
                If None, will be inferred from model layer configuration.
                Defaults to None.
            gpu_idx (typing.List[int], optional): GPUs to train on.
                If empty, model will be trained on CPU.
                Defaults to [].
            early_stopping (int, optional): Patience for early stopping.
                If None, no early stopping will be applied.
                Defaults to None.

        Returns:
            AEModel: the model.
        """
        log('Reading data...', verbose=verbose)
        train_X, train_Y = utils.load_npy_dataset(os.path.join(
            self.workdir_path, 'data', 'train'))
        val_X, val_Y = utils.load_npy_dataset(os.path.join(
            self.workdir_path, 'data', 'val'))
        train_X = np.array(train_X, dtype='float32')
        train_Y = np.array(train_Y, dtype='float32')
        val_X = np.array(val_X, dtype='float32')
        val_Y = np.array(val_Y, dtype='float32')
        # n_features = the number of columns in dataset
        n_features = train_X.shape[1]
        # setup the environment -- gpu or cpu
        device_name = utils.setup_torch_environment(
            gpus_idx=gpu_idx, verbose=verbose)
        device = torch.device(device_name)

        if bottleneck_size is None:
            bottleneck_size = n_features // 2
        log('Bulding model...', verbose=verbose)
        self.model = AEModel(
            layer_config=layer_configs,
            bottleneck_size=bottleneck_size,
            in_size=train_X.shape[1],
            out_size=train_X.shape[1]
        )

        self._optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=learning_rate
        )
        # save the number of expochs this model was trained in
        self.model._n_epochs = epochs

        if (return_compiled_model is True):
            log(f'Cancel training, return compiled model...', verbose=verbose)
            return self.model

        # where to save the model
        if (model_path is None):
            model_path = self.get_model_path_from_layer_config()
        # prepare dataloaders for torch
        trainx_loader = torch.utils.data.DataLoader(
            torch.tensor(train_X), batch_size=batch_size, shuffle=False)
        trainy_loader = torch.utils.data.DataLoader(
            torch.tensor(train_Y), batch_size=batch_size, shuffle=False)
        valx_loader = torch.utils.data.DataLoader(
            torch.tensor(val_X), batch_size=batch_size, shuffle=False)
        valy_loader = torch.utils.data.DataLoader(
            torch.tensor(val_Y), batch_size=batch_size, shuffle=False)
        log(f'Moving everything to device {device_name}', verbose=True)
        self.model = self.model.to(device)
        log(self.model, verbose=verbose)

        from math import ceil
        percentile10 = ceil(epochs / 10)
        self.model.train_losses = []
        self.model.val_losses = []
        best_epoch_idx = 0
        for _epoch_idx in range(0, epochs):
            # for each epoch, do...:
            # switch model to training mode
            self.model.train()
            train_epoch_loss = self.fit_epoch(
                self.model, self._optimizer,
                trainx_loader, trainy_loader, device, False)
            self.model.train_losses.append(
                train_epoch_loss.cpu().detach().numpy()
            )

            # and now val mode
            self.model.eval()
            with torch.no_grad():
                val_epoch_loss = self.fit_epoch(
                    self.model, self._optimizer,
                    valx_loader, valy_loader, device, True)
                self.model.val_losses.append(
                    # if it's on gpu, this will transfer it onto cpu
                    val_epoch_loss.cpu().detach().numpy()
                )
            # when every 10% of epoch passses, print the logs
            log(
                f'Epoch: {(_epoch_idx+1):5} / {epochs};' +
                f' train loss: {train_epoch_loss:.5f},' +
                f' val loss: {val_epoch_loss:.5f}',
                end='\n' if (_epoch_idx+1) % percentile10 == 0 else '\r',
                verbose=verbose
            )
            # check if this epoch is the best
            last_val_loss = self.model.val_losses[-1]
            best_val_loss = self.model.val_losses[best_epoch_idx]
            if (last_val_loss <= best_val_loss):
                best_epoch_idx = _epoch_idx
                # save model
                utils.save_torch_model(self.model, optimizer=self._optimizer,
                                       epoch=_epoch_idx, path=model_path,
                                       verbose=verbose, best=True)

            # check if we should stop training (early stopping)
            if (early_stopping is not None and
                    _epoch_idx - best_epoch_idx > early_stopping):
                log(f'Early stopping at epoch {_epoch_idx}.' +
                    f'Best epoch was {best_epoch_idx} ' +
                    f'with early stopping patience {early_stopping}.',
                    verbose=verbose)
                # if this model was stopped by early stopping,
                # then adjust n_epochs
                self.model._n_epochs = best_epoch_idx + 1
                break

        best_val_loss = self.model.val_losses[best_epoch_idx]
        best_train_loss = self.model.train_losses[best_epoch_idx]
        log(
            f'Train loss at best epoch: {best_train_loss} & ',
            f'Val loss at best epoch: {best_val_loss}',
            verbose=verbose
        )

        fig, axes = plt.subplots(1, 1, constrained_layout=True)
        fig.suptitle(
            f'Train loss at best epoch: {best_train_loss} \n' +
            f'Val loss at best epoch: {best_val_loss}')
        axes.plot(list(range(_epoch_idx+1)),
                  self.model.train_losses, label='Train loss')
        axes.plot(list(range(_epoch_idx+1)),
                  self.model.val_losses, label='Val loss')
        axes.axvline(x=best_epoch_idx, label='Best Epoch')
        axes.legend()
        plt.savefig(f'{model_path}.png')

        # after the model was trained, restore the best weights
        self.model, self._optimizer = utils.load_torch_model(
            model=self.model,
            optimizer=self._optimizer,
            path=model_path,
            best=True
        )

        return self.model

    def get_model_path_from_layer_config(
        self,
        layer_configs: typing.List[LayerConfig] = None,
        bottleneck_size: float = None,
        learning_rate: float = None,
        epochs: int = None
    ):
        """Retrieve path to model based on
        parameters of model.

        Args:
            layer_configs (typing.List[LayerConfig], optional):
                Configuration of the model layers.
                If None, will be inferred from class-attribute `model`.
                Defaults to None.
            bottleneck_size (float, optional):
                Size of the bottleneck layer.
                If None, will be inferred from class-attribute `model`.
                Defaults to None.
            learning_rate (float, optional):
                Learning rate of the optimizer.
                If None, will be inferred from class-attribute `model`.
                Defaults to None.
            epochs (int, optional): number of epochs the model was trained on.
                For now, this is ignored.
                Defaults to None.

        Returns:
            str: Path to the model, without extension.
        """
        if layer_configs is None:
            layer_configs = self.model._layer_config
        if bottleneck_size is None:
            bottleneck_size = self.model._bottleneck_size
        if epochs is None:
            epochs = self.model._n_epochs
        if learning_rate is None:
            learning_rate = self._optimizer.param_groups[0]['lr']
        model_desc = ''
        for layer in layer_configs:
            model_desc += f'{layer.type}-{layer.size}-{layer.dropout}_'
        model_desc += f'_bottle-{bottleneck_size}_lr-{learning_rate}'
        # for now, we don't care about n_epochs
        # model_desc += f'_epochs-{epochs}'
        model_path = os.path.join(
            self.workdir_path,
            'models',
            f'{model_desc}',
        )
        return model_path

    def cluster_and_get_metrics(
        self, algorithm: str,
        # df containing data from validation dataset
        original_df: pd.DataFrame,
        list_n_clusters: typing.List[int],
        verbose: bool = False,
        random_state: typing.List[int] = 1,
    ) -> pd.DataFrame:
        """Get latent representations of the data using the model
        And then clister the compressed data and obtain results.

        Args:
            algorithm (str): Which algorithm to use.
            original_df (pd.DataFrame): Dataframe containing original,
                unencoded data.
            list_n_clusters (typing.List[int]): A list of
                number of clusters to test.
            verbose (bool, optional): Print useful logs.
                Defaults to False.
            random_state (int, optional): For reproducibilty.
                Defaults to 1.

        Returns:
            pd.DataFrame: Dataframe containing the results.
        """
        # gather data for clustering
        train_data, _ = utils.load_npy_dataset(os.path.join(
            self.workdir_path, 'data', 'train'))
        test_data, _ = utils.load_npy_dataset(os.path.join(
            self.workdir_path, 'data', 'test'))
        train_data = np.array(train_data, dtype='float32')
        test_data = np.array(test_data, dtype='float32')
        # transfer model to CPU to get latent rep
        self.model.cpu()
        # evaluation mode -- we don't want to update weights
        self.model.eval()
        # get latent/compressed version of data
        log('Transferring into latent representation,' +
            f' from shape {np.shape(train_data)}',
            verbose=verbose)
        train_data, _ = self.model.get_latent(torch.tensor(train_data))
        train_data: np.ndarray = train_data.detach().numpy()
        test_data, _ = self.model.get_latent(torch.tensor(test_data))
        test_data: np.ndarray = test_data.detach().numpy()
        log('Transferred into latent representation,' +
            f' to shape {np.shape(train_data)}',
            verbose=verbose)
        from radiologynet.tools.metrics.clustering import \
            cluster_and_get_metrics
        _p = self.get_model_path_from_layer_config()
        _p = f'{_p}-clustermodels'

        results = cluster_and_get_metrics(
            algorithm=algorithm,
            train_data=train_data,
            test_data=test_data,
            original_df=original_df,
            verbose=verbose,
            random_state=random_state,
            list_n_clusters=list_n_clusters,
            clusterer_save_dir=_p
        )
        name = os.path.basename(self.get_model_path_from_layer_config())
        results['Name'] = name
        return results
