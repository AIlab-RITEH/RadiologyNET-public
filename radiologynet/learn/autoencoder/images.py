import typing

from radiologynet.learn.autoencoder.dicom_tags import LayerConfig
import tensorflow as tf
from radiologynet.logging import log
from radiologynet.tools.image_conversion import basic as basic_img_io
import os
import shutil
import numpy as np
from radiologynet.learn.utils import load_npy_dataset, to_npy_dataset
from radiologynet.tools.visualization import \
    training_stats as plt_training_stats
import matplotlib.pyplot as plt


class DicomImageConvolutionAutoEncoder():
    def __init__(
        self,
        workdir_path: str,
        images_path: str,
        layer_configs: typing.List[LayerConfig] = [],
        bottleneckcfg: LayerConfig = None,
        learning_rate: float = 0.01,
        batch_size=2**10,
        epochs=200,

    ) -> None:
        self.workdir_path = workdir_path
        self.images_path = images_path
        self.layer_configs = layer_configs
        self.bottleneckcfg = bottleneckcfg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def preprocess_images(
        self,
        ids: typing.List[int],
    ):
        """Perform image preprocessing
        and preparation for CAE.

        This will:
            - load 8-bit PNG image from file system
            - scale it into the range [0, 1] by diving with 255
            - expand dims so what grescale images will be, e.g.
                (256, 256, 1)
                instead of (256, 256)

        Args:
            ids (typing.List[int]): DCM IDs of images
            which should be prepared for CAE.

        Returns:
            np.ndarray: a npy array containing all of the images.
                This is a 4D array (e.g., 100 images of dims 256x256x1).
        """
        imgs = []
        for dcm_id in ids:
            img = basic_img_io.load_converted_image(
                self.images_path,
                dcm_id,
                extension='png',
                return_2d=True
            )
            # img = img.resize((128, 128))
            img = np.expand_dims(img, axis=2)
            img = img / 255.
            imgs.append(img)
        imgs = np.array(imgs)
        return imgs

    def to_npy_dataset(
        self,
        train_ids: typing.List[int],
        test_ids: typing.List[int],
        val_ids: typing.List[int],
        save_to: str = None,
        verbose: bool = False,
    ):
        """Preprocess images and save them as npy datasets.

        Args:
            train_ids (typing.List[int]): DCM IDs of images
                which should be used for training.
            test_ids (typing.List[int]): DCM IDs of images
                which should be used for testing.
            val_ids (typing.List[int]): DCM IDs of images
                which should be used for validation.
            save_to (str, optional): Where to save the npy datasets.
                If None, will be saved to `workdir_path`.
                Defaults to None.
            verbose (bool, optional): Whether to print useful logs.
                Defaults to False.
        """
        if (save_to is None):
            save_to = self.workdir_path

        dataset_meta = zip([train_ids, test_ids, val_ids],
                           ['train', 'test', 'val'])

        for ids, ds_name in dataset_meta:
            path = os.path.join(save_to, 'data', f'{ds_name}')
            # if this path already exists, delete it
            # because who knows what's in there
            if os.path.exists(path):
                log(f'Found something at {path}. Deleting it...',
                    verbose=verbose)
                shutil.rmtree(path)

            log(f'Parsing images for dataset "{ds_name}"')
            imgs = self.preprocess_images(ids=ids)

            to_npy_dataset(X=imgs, Y=imgs, save_to=path)
            log(f'Saved dataset "{ds_name}" to {path}', verbose=verbose)

    def build_model(
        self,
        layer_configs: typing.List[LayerConfig],
        input_image_shape: typing.Tuple,
        bottleneck: LayerConfig,
        verbose: bool = False
    ):
        # step 1: define the input.
        log(
            f'Bulding network with input shape: {input_image_shape}',
            verbose=verbose
        )
        cur_img_shape = np.array(input_image_shape)

        def add_layers(
            lyr,
            layer_configs=layer_configs,
            upsample: bool = False,
            cur_img_shape=cur_img_shape
        ):
            for layer in layer_configs:
                if(layer.type == 'Conv'):
                    lyr = tf.keras.layers.Conv2D(
                        layer.nfilters,
                        layer.kernel_size,
                        padding='same',
                        activation='relu'
                    )(lyr)
                elif(layer.type == 'MaxPool'):
                    if(upsample is True):
                        # decoder part --> upsample
                        lyr = tf.keras.layers.UpSampling2D(
                            layer.kernel_size
                        )(lyr)
                        cur_img_shape = cur_img_shape * layer.kernel_size

                    else:
                        # encoder part --> downsample
                        lyr = tf.keras.layers.MaxPooling2D(
                            layer.kernel_size,
                            strides=layer.kernel_size
                        )(lyr)
                        cur_img_shape = cur_img_shape / layer.kernel_size

            return lyr, cur_img_shape

        inputs = tf.keras.layers.Input(shape=input_image_shape)

        # ENCODER
        encoder, cur_img_shape = add_layers(inputs)

        # # BOTTLENECK
        bottlenecklyr = encoder
        # flatten the layers to get a 1d array
        bottlenecklyr = tf.keras.layers.Flatten()(bottlenecklyr)
        # squeeze the 1d array into the bottleneck layer
        bottlenecklyr = tf.keras.layers.Dense(
            bottleneck.nfilters, name='BOTTLENECK'
        )(bottlenecklyr)
        decoder = bottlenecklyr
        # figure out the dimensions of the layer which is unsqueezed
        reshape_to = (
            int(cur_img_shape[0]), int(
                cur_img_shape[1]), layer_configs[-2].nfilters
        )
        # DECODER
        # this is the decoder beginning
        # unsqueeze the bottleneck layer as part of the decoder
        decoder = tf.keras.layers.Dense(
            reshape_to[0] * reshape_to[1] * reshape_to[2]
        )(decoder)
        decoder = tf.keras.layers.Reshape(reshape_to)(decoder)

        decoder, cur_img_shape = add_layers(
            decoder,
            reversed(layer_configs),
            upsample=True
        )

        outputs = tf.keras.layers.Conv2D(
            input_image_shape[-1], 3, padding="same", activation='sigmoid'
        )(decoder)

        model = tf.keras.Model(inputs, outputs, name="CAE")

        if(verbose is True):
            print(model.summary())
        return model, bottlenecklyr, inputs

    def train(
        self,
        return_compiled_model: bool = False,
        verbose: bool = False
    ):
        layer_configs = self.layer_configs
        bottleneckcfg = self.bottleneckcfg
        learning_rate = self.learning_rate
        batch_size = self.batch_size
        epochs = self.epochs

        train_X, train_Y = load_npy_dataset(os.path.join(
            self.workdir_path, 'data', 'train'))
        val_X, val_Y = load_npy_dataset(os.path.join(
            self.workdir_path, 'data', 'val'))

        # at position [0] in element_spec are the X values
        # at position [1] are the true labels (in other words - Y)
        input_image_shape = np.shape(train_X[0])

        model, bottleneck, inputs = self.build_model(
            input_image_shape=input_image_shape,
            layer_configs=layer_configs,
            verbose=verbose,
            bottleneck=bottleneckcfg
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanSquaredError()
        accuracy = tf.keras.metrics.Accuracy()
        model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

        if (return_compiled_model is True):
            log(f'Cancel training, return compiled model...', verbose=verbose)
            return model, bottleneck, inputs

        EARLY_STOPPING_PATIENCE = 20
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=verbose,
            patience=EARLY_STOPPING_PATIENCE,
            mode='min',
            restore_best_weights=True
        )

        # where to save the model
        model_path = self.get_model_path_from_layer_config()

        # this saves the entire autoencoder for later evaluation
        checkpoint_filepath = f'{model_path}__checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=verbose
        )

        history = model.fit(
            x=train_X,
            y=train_Y,
            epochs=epochs,
            validation_data=(val_X, val_Y),
            callbacks=[early_stopping, model_checkpoint_callback],
            verbose=verbose,
            batch_size=batch_size
        )
        log(
            f'Last val accuracy: {history.history["val_accuracy"][-1]}',
            verbose=verbose
        )
        best_epoch = early_stopping.best_epoch
        best_val_acc = history.history['val_accuracy'][best_epoch]
        log(f'Best val accuracy: {best_val_acc}', verbose=verbose)

        fig, ax = plt_training_stats.plot_ml_history_stats(
            history, early_stopping)
        fig.suptitle(
            f'Best Val Accuracy: {best_val_acc:.4f}\nBest Epoch: {best_epoch}')

        plt.savefig(f'{model_path}.png')

        # save the encoder for use later
        encoder = tf.keras.models.Model(inputs=inputs, outputs=bottleneck)
        encoder.compile(loss=loss, metrics='accuracy')

        log(f'Saving model to "{model_path}"')
        encoder.save(model_path)

        return model

    def get_model_path_from_layer_config(self):
        layer_configs = self.layer_configs
        bottleneck_size = self.bottleneckcfg.nfilters
        learning_rate = self.learning_rate

        model_desc = ''
        for layer in layer_configs:
            model_desc += f'{layer.type}-{layer.nfilters}-{layer.kernel_size}_'
        model_desc += f'_bottle-{bottleneck_size}_lr-{learning_rate}'
        model_path = os.path.join(
            self.workdir_path,
            'models',
            f'{model_desc}.h5',
        )
        return model_path
