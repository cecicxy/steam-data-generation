"""以下是noise 的函数（自己修改的，改成了一般正态分布，参数可调）"""


DGAN.attribute_noise_func = lambda batch_size: self.config.attribute_noise_std*torch.randn(
            batch_size, self.config.attribute_noise_dim, device=self.device
        )+self.config.attribute_noise_bias


DGAN.feature_noise_func = lambda batch_size: self.config.feature_noise_std*torch.randn(
            batch_size,
            self.config.max_sequence_len // self.config.sample_len,
            self.config.feature_noise_dim,
            device=self.device,
        )+self.config.feature_noise_bias
##### 这里self.config 都在下面有个config 类里有解释
DGAN.generator = Generator(
            attribute_outputs,
            self.additional_attribute_outputs,
            feature_outputs,
            self.config.max_sequence_len,
            self.config.sample_len,
            self.config.attribute_noise_dim,
            self.config.feature_noise_dim,
            self.config.attribute_num_units,
            self.config.attribute_num_layers,
            self.config.feature_num_units,
            self.config.feature_num_layers,
        )
## 定义在下面具体结构中
DGAN.feature_discriminator = Discriminator(
            attribute_dim
            + additional_attribute_dim
            + self.config.max_sequence_len * feature_dim,
            num_layers=5,
            num_units=200,
        )

### 同样，在下面

DGAN.attribute_discriminator = Discriminator(
                attribute_dim + additional_attribute_dim,
                num_layers=5,
                num_units=200,
            )

'''CONFIG 类'''
class DGANConfig:
    """Config object with parameters for training a DGAN model.

    Args:
        max_sequence_len: length of time series sequences, variable length
            sequences are not supported, so all training and generated data will
            have the same length sequences
        sample_len: time series steps to generate from each LSTM cell in DGAN,
            must be a divisor of max_sequence_len
        attribute_noise_dim: length of the GAN noise vectors for attribute
            generation
        feature_noise_dim: length of GAN noise vectors for feature generation
        attribute_num_layers: # of layers in the GAN discriminator network
        attribute_num_units: # of units per layer in the GAN discriminator
            network
        feature_num_layers: # of LSTM layers in the GAN generator network
        feature_num_units: # of units per layer in the GAN generator network
        use_attribute_discriminator: use separaste discriminator only on
            attributes, helps DGAN match attribute distributions, Default: True
        normalization: default normalization for continuous variables, used when
            metadata output is not specified during DGAN initialization
        apply_feature_scaling: scale each continuous variable to [0,1] or [-1,1]
            (based on normalization param) before training and rescale to
            original range during generation, if False then training data must
            be within range and DGAN will only generate values in [0,1] or
            [-1,1], Default: True
        apply_example_scaling: compute midpoint and halfrange (equivalent to
            min/max) for each time series variable and include these as
            additional attributes that are generated, this provides better
            support for time series with highly variable ranges, e.g., in
            network data, a dial-up connection has bandwidth usage in [1kb,
            10kb], while a fiber connection is in [100mb, 1gb], Default: True
        binary_encoder_cutoff: use binary encoder (instead of one hot encoder) for
            any column with more than this many unique values. This helps reduce memory
            consumption for datasets with a lot of unique values.
        forget_bias: initialize forget gate bias paramters to 1 in LSTM layers,
            when True initialization matches tf1 LSTMCell behavior, otherwise
            default pytorch initialization is used, Default: False
        gradient_penalty_coef: coefficient for gradient penalty in Wasserstein
            loss, Default: 10.0
        attribute_gradient_penalty_coef: coefficient for gradient penalty in
            Wasserstein loss for the attribute discriminator, Default: 10.0
        attribute_loss_coef: coefficient for attribute discriminator loss in
            comparison the standard discriminator on attributes and features,
            higher values should encourage DGAN to learn attribute
            distributions, Default: 1.0
        generator_learning_rate: learning rate for Adam optimizer
        generator_beta1: Adam param for exponential decay of 1st moment
        discriminator_learning_rate: learning rate for Adam optimizer
        discriminator_beta1: Adam param for exponential decay of 1st moment
        attribute_discriminator_learning_rate: learning rate for Adam optimizer
        attribute_discriminator_beta1: Adam param for exponential decay of 1st
            moment
        batch_size: # of examples used in batches, for both training and
            generation
        epochs: # of epochs to train model discriminator_rounds: training steps
        for the discriminator(s) in each
            batch
        generator_rounds: training steps for the generator in each batch
        cuda: use GPU if available
        mixed_precision_training: enabling automatic mixed precision while training
            in order to reduce memory costs, bandwith, and time by identifying the
            steps that require full precision and using 32-bit floating point for
            only those steps while using 16-bit floating point everywhere else.
    """
    # Model structure
    max_sequence_len: int
    sample_len: int
    #### 以下是默认参数
    attribute_noise_dim: int = 10
    feature_noise_dim: int = 10
    attribute_noise_std: float = 1
    feature_noise_std: float = 0.3
    feature_noise_bias: float = 0.5
    attribute_noise_bias: float = 0
    attribute_num_layers: int = 3
    attribute_num_units: int = 100
    feature_num_layers: int = 1
    feature_num_units: int = 100
    use_attribute_discriminator: bool = True

    # Data transformation
    normalization: Normalization = Normalization.ZERO_ONE
    apply_feature_scaling: bool = True
    apply_example_scaling: bool = True
    binary_encoder_cutoff: int = 150

    # Model initialization
    forget_bias: bool = False

    # Loss function
    gradient_penalty_coef: float = 10.0
    attribute_gradient_penalty_coef: float = 10.0
    attribute_loss_coef: float = 1.0

    # Training
    generator_learning_rate: float = 0.001
    generator_beta1: float = 0.5
    discriminator_learning_rate: float = 0.001
    discriminator_beta1: float = 0.5
    attribute_discriminator_learning_rate: float = 0.001
    attribute_discriminator_beta1: float = 0.5
    batch_size: int = 1024
    epochs: int = 400
    discriminator_rounds: int = 1
    generator_rounds: int = 1

    cuda: bool = True
    mixed_precision_training: bool = False





"""以下是DGAN的具体结构"""
"""Internal module with torch implementation details of DGAN."""

from collections import OrderedDict
from typing import cast, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from gretel_synthetics.timeseries_dgan.config import Normalization
from gretel_synthetics.timeseries_dgan.transformations import (
    BinaryEncodedOutput,
    ContinuousOutput,
    OneHotEncodedOutput,
    Output,
)


class Merger(torch.nn.Module):
    """Merge several torch layers with same inputs into one concatenated layer."""

    def __init__(
        self,
        modules: Union[torch.nn.ModuleList, Iterable[torch.nn.Module]],
        dim_index: int,
    ):
        """Create Merge module that concatenates layers.

        Args:
            modules: modules (layers) to merge
            dim_index: dim for the torch.cat operation, often the last dimension
                of the tensors involved
        """
        super(Merger, self).__init__()
        if isinstance(modules, torch.nn.ModuleList):
            self.layers = modules
        else:
            self.layers = torch.nn.ModuleList(modules)

        self.dim_index = dim_index

    def forward(self, input):
        """Apply module to input.

        Args:
            input: whatever the layers are expecting, usually a Tensor or tuple
                of Tensors

        Returns:
            Concatenation of outputs from layers.
        """
        return torch.cat([m(input) for m in self.layers], dim=self.dim_index)


class OutputDecoder(torch.nn.Module):
    """Decoder to produce continuous or discrete output values as needed."""

    def __init__(self, input_dim: int, outputs: List[Output], dim_index: int):
        """Create decoder to make final output for a variable in DGAN.

        Args:
            input_dim: dimension of input vector
            outputs: list of variable metadata objects to generate
            dim_index: dim for torch.cat operation, often the last dimension
                of the tensors involved
        """
        super(OutputDecoder, self).__init__()
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("OutputDecoder received no outputs")

        self.dim_index = dim_index
        self.generators = torch.nn.ModuleList()

        for output in outputs:
            if "OneHotEncodedOutput" in str(output.__class__):
                output = cast(OneHotEncodedOutput, output)
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(int(input_dim), int(output.dim)),
                                ),
                                ("softmax", torch.nn.Softmax(dim=int(dim_index))),
                            ]
                        )
                    )
                )
            elif "BinaryEncodedOutput" in str(output.__class__):
                output = cast(BinaryEncodedOutput, output)
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(int(input_dim), int(output.dim)),
                                ),
                                (
                                    "sigmoid",
                                    torch.nn.Sigmoid(),
                                ),
                            ]
                        )
                    )
                )
            elif "ContinuousOutput" in str(output.__class__):
                output = cast(ContinuousOutput, output)
                if output.normalization == Normalization.ZERO_ONE:
                    normalizer = torch.nn.Sigmoid()
                elif output.normalization == Normalization.MINUSONE_ONE:
                    normalizer = torch.nn.Tanh()
                else:
                    raise RuntimeError(
                        f"Unsupported normalization='{output.normalization}'"
                    )
                self.generators.append(
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "linear",
                                    torch.nn.Linear(int(input_dim), int(output.dim)),
                                ),
                                ("normalization", normalizer),
                            ]
                        )
                    )
                )
            else:
                raise RuntimeError(f"Unsupported output type, class={type(output)}'")

    def forward(self, input):
        """Apply module to input.

        Args:
            input: tensor with last dim of size input_dim

        Returns:
            Generated variables packed into a single tensor (in same order as outputs).
        """
        outputs = [generator(input) for generator in self.generators]
        merged = torch.cat(outputs, dim=self.dim_index)
        return merged


class SelectLastCell(torch.nn.Module):
    """Select just the last layer's hidden output from LSTM module."""

    def forward(self, x):
        """Apply module to input.

        Args:
            x: tensor output from an LSTM layer

        Returns:
            Tensor of last layer hidden output.
        """
        out, _ = x
        return out
# 该类名为SelectLastCell，它是一个简单的自定义模块。该类的作用是从LSTM模块的输出张量中仅选择最后一层的隐藏输出作为模型的输出。

# 类中只有一个函数forward(self, x)，表示前向传递。输入为LSTM层的输出张量（通常包含隐藏层和单元状态），通过将其分为两个输出变量（最后一层的隐藏输出和单元状态）来仅保留最后一层的隐藏输出。之后，仅返回最后一层的隐藏输出作为模型的输出。

class Generator(torch.nn.Module):
    """Generator networks for attributes and features of DGAN model."""

    def __init__(
        self,
        attribute_outputs: Optional[List[Output]],
        additional_attribute_outputs: Optional[List[Output]],
        feature_outputs: List[Output],
        max_sequence_len: int,
        sample_len: int,
        attribute_noise_dim: Optional[int],
        feature_noise_dim: int,
        attribute_num_units: Optional[int],
        attribute_num_layers: Optional[int],
        feature_num_units: int,
        feature_num_layers: int,
    ):
        """Create generator network.

        Args:
            attribute_outputs: metadata objects for attribute variables to
                generate
            additional_attribute_outputs: metadata objects for additional
                attribute variables to generate
            feature_outputs: metadata objects for feature variables to generate
            max_sequence_len: length of feature time sequences
            sample_len: # of time points to generate from each LSTM cell
            attribute_noise_dim: size of noise vector for attribute GAN
            feature_noise_dim: size of noise vector for feature GAN
            attribute_num_units: # of units per layer in MLP used to generate
                attributes
            attribute_num_layers: # of layers in MLP used to generate attributes
            feature_num_units: # of units per layer in LSTM used to generate
                features
            feature_num_layers: # of layers in LSTM used to generate features
        """

# 这些参数分别表示要生成的属性变量、附加属性变量和特征变量的元数据对象；特征时间序列的长度；每个 LSTM 单元要生成的时间点数；属性 GAN 和特征 GAN 的噪声向量大小；用于生成属性和特征的 MLP 和 LSTM 的每层单元数和层数。


        super(Generator, self).__init__()
        assert max_sequence_len % sample_len == 0

        self.sample_len = sample_len
        self.max_sequence_len = max_sequence_len
        self.attribute_gen, attribute_dim = self._make_attribute_generator(
            attribute_outputs,
            attribute_noise_dim,
            attribute_num_units,
            attribute_num_layers,
        )
        (
            self.additional_attribute_gen,
            additional_attribute_dim,
        ) = self._make_attribute_generator(
            additional_attribute_outputs,
            attribute_noise_dim + attribute_dim,
            attribute_num_units,
            attribute_num_layers,
        )
        self.feature_gen = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "lstm",
                        torch.nn.LSTM(
                            int(
                                attribute_dim
                                + additional_attribute_dim
                                + feature_noise_dim
                            ),
                            int(feature_num_units),
                            int(feature_num_layers),
                            batch_first=True,
                        ),
                    ),
                    ("selector", SelectLastCell()),
                    (
                        "merger",
                        Merger(
                            [
                                OutputDecoder(
                                    int(feature_num_units), feature_outputs, dim_index=2
                                )
                                for _ in range(self.sample_len)
                            ],
                            dim_index=2,
                        ),
                    ),
                ]
            )
        )

    def _make_attribute_generator(
        self, outputs: List[Output], input_dim: int, num_units: int, num_layers: int
    ) -> torch.nn.Sequential:
        """Helper function to create generator network for attributes.

        Used to build the generater for both the attribute and additional
        attribute generation. The output dimension of the newly built
        generator is also outputted. This is useful when passing these
        attributes into other generators.

        Args:
            outputs: metadata objects for variables
            input_dim: size of input vectors (usually random noise)
            num_units: # of units per layer in MLP
            num_layers: # of layers in MLP

        Returns:
            Feed-forward MLP to generate attributes, wrapped in a
            torch.nn.Sequential module.
            Attribute dimension for LSTM layer size in generator.
        """
        if not outputs:
            return None, 0
        seq = []
        last_dim = int(input_dim)
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(int(last_dim), int(num_units)))
            seq.append(torch.nn.ReLU())
            seq.append(torch.nn.BatchNorm1d(int(num_units)))
            last_dim = int(num_units)

        seq.append(OutputDecoder(int(last_dim), outputs, dim_index=1))
        attribute_dim = sum(output.dim for output in outputs)
        return torch.nn.Sequential(*seq), int(attribute_dim)

    def forward(
        self, attribute_noise: torch.Tensor, feature_noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply module to input.

        Args:
            attribute_noise: noise tensor for attributes, 2d tensor of (batch
                size, attribute_noise_dim) shape
            feature_noise: noise tensor for features, 3d tensor of (batch size,
                max_sequence_len, feature_noise_dim) shape

        Returns:
            Tuple of generated tensors with attributes (if present), additional_attributes
            (if present), and features. The tuple is structured as follows: (attributes,
            additional_attributes, features). If attributes and/or additional_attributes is not
            present, an empty nan-filled tensor will be returned in the tuple. The function
            will always return a 3-element tensor tuple.
        """

        # Attribute features exist

        empty_tensor = torch.Tensor(np.full((1, 1), np.nan))

        if self.attribute_gen is not None:
            attributes = self.attribute_gen(attribute_noise)

            if self.additional_attribute_gen:
                # detach() should be equivalent to stop_gradient used in tf1 code.
                attributes_no_gradient = attributes.detach()
                additional_attribute_gen_input = torch.cat(
                    (attributes_no_gradient, attribute_noise), dim=1
                )

                additional_attributes = self.additional_attribute_gen(
                    additional_attribute_gen_input
                )
                combined_attributes = torch.cat(
                    (attributes, additional_attributes), dim=1
                )
            else:
                additional_attributes = empty_tensor
                combined_attributes = attributes

            # Use detach() to stop gradient flow
            combined_attributes_no_gradient = combined_attributes.detach()

            reshaped_attributes = torch.reshape(
                combined_attributes_no_gradient, (combined_attributes.shape[0], 1, -1)
            )
            reshaped_attributes = reshaped_attributes.expand(
                -1, feature_noise.shape[1], -1
            )

            feature_gen_input = torch.cat((reshaped_attributes, feature_noise), 2)

            features = self.feature_gen(feature_gen_input)

            features = torch.reshape(
                features, (features.shape[0], self.max_sequence_len, -1)
            )
            return attributes, additional_attributes, features
        else:

            if self.additional_attribute_gen:
                additional_attributes = self.additional_attribute_gen(attribute_noise)
                combined_attributes_no_gradient = additional_attributes.detach()
                reshaped_attributes = torch.reshape(
                    combined_attributes_no_gradient,
                    (additional_attributes.shape[0], 1, -1),
                )
                reshaped_attributes = reshaped_attributes.expand(
                    -1, feature_noise.shape[1], -1
                )
                feature_gen_input = torch.cat((reshaped_attributes, feature_noise), 2)
                features = self.feature_gen(feature_gen_input)
                features = torch.reshape(
                    features, (features.shape[0], self.max_sequence_len, -1)
                )
                return empty_tensor, additional_attributes, features

            else:
                features = self.feature_gen(feature_noise)
                features = torch.reshape(
                    features, (features.shape[0], self.max_sequence_len, -1)
                )
                return empty_tensor, empty_tensor, features


class Discriminator(torch.nn.Module):
    """Discriminator network for DGAN model."""

    def __init__(self, input_dim: int, num_layers: int = 5, num_units: int = 200):
        """Create discriminator network.

        Args:
            input_dim: size of input to discriminator network
            num_layers: # of layers in MLP used for discriminator
            num_units: # of units per layer in MLP used for discriminator
        """
        super(Discriminator, self).__init__()

        seq = []
        last_dim = input_dim
        for _ in range(num_layers):
            seq.append(torch.nn.Linear(int(last_dim), int(num_units)))
            seq.append(torch.nn.ReLU())
            last_dim = num_units

        seq.append(torch.nn.Linear(int(last_dim), 1))

        self.seq = torch.nn.Sequential(*seq)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply module to input.

        Args:
            input: input tensor of shape (batch size, input_dim)

        Returns:
            Discriminator output with shape (batch size, 1).
        """
        return self.seq(input)


'''训练过程'''

def _train(
    self,
    dataset: Dataset,
):
        """Internal method for training DGAN model.

        Expects data to already be transformed into the internal representation
        and wrapped in a torch Dataset. The torch Dataset consists of 3-element
        tuples (attributes, additional_attributes, features). If attributes and/or
        additional_attribtues were not passed by the user, these indexes of the
        tuple will consists of nan-filled tensors which will later be filtered
        out and ignored in the DGAN training process.

        Args:
            dataset: torch Dataset containing tuple of (attributes, additional_attributes, features)
        """
        if len(dataset) <= 1:
            raise ValueError(
                f"DGAN requires multiple examples to train, received {len(dataset)} example."
                + "Consider splitting a single long sequence into many subsequences to obtain "
                + "multiple examples for training."
            )

        # Our optimization setup does not work on batches of size 1. So if
        # drop_last=False would produce a last batch of size of 1, we use
        # drop_last=True instead.
        drop_last = len(dataset) % self.config.batch_size == 1

        loader = DataLoader(
            dataset,
            self.config.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=0,
            # prefetch_factor=4,
            # persistent_workers=True,
            # pin_memory=True,
            # multiprocessing_context="fork",
        )

        opt_discriminator = torch.optim.Adam(
            self.feature_discriminator.parameters(),
            lr=self.config.discriminator_learning_rate,
            betas=(self.config.discriminator_beta1, 0.999),
        )

        opt_attribute_discriminator = None
        if self.attribute_discriminator is not None:
            opt_attribute_discriminator = torch.optim.Adam(
                self.attribute_discriminator.parameters(),
                lr=self.config.attribute_discriminator_learning_rate,
                betas=(self.config.attribute_discriminator_beta1, 0.999),
            )

        opt_generator = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.generator_learning_rate,
            betas=(self.config.generator_beta1, 0.999),
        )

        global_step = 0

        # Set torch modules to training mode
        self._set_mode(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision_training)

        for epoch in range(self.config.epochs):

            for batch_idx, real_batch in enumerate(loader):
                global_step += 1

                with torch.cuda.amp.autocast(
                    enabled=self.config.mixed_precision_training
                ):
                    attribute_noise = self.attribute_noise_func(real_batch[0].shape[0])
                    feature_noise = self.feature_noise_func(real_batch[0].shape[0])

                    # Both real and generated batch are always three element tuple of
                    # tensors. The tuple is structured as follows: (attribute_output,
                    # additional_attribute_output, feature_output). If self.attribute_output
                    # and/or self.additional_attribute_output is empty, the respective
                    # tuple index will be filled with a placeholder nan-filled tensor.
                    # These nan-filled tensors get filtered out in the _discriminate,
                    # _get_gradient_penalty, and _discriminate_attributes functions.

                    generated_batch = self.generator(attribute_noise, feature_noise)
                    real_batch = [
                        x.to(self.device, non_blocking=True) for x in real_batch
                    ]

                for _ in range(self.config.discriminator_rounds):
                    opt_discriminator.zero_grad(
                        set_to_none=self.config.mixed_precision_training
                    )
                    with torch.cuda.amp.autocast(enabled=True):
                        generated_output = self._discriminate(generated_batch)
                        real_output = self._discriminate(real_batch)

                        loss_generated = torch.mean(generated_output)
                        loss_real = -torch.mean(real_output)
                        loss_gradient_penalty = self._get_gradient_penalty(
                            generated_batch, real_batch, self._discriminate
                        )

                        loss = (
                            loss_generated
                            + loss_real
                            + self.config.gradient_penalty_coef * loss_gradient_penalty
                        )

                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(opt_discriminator)
                    scaler.update()

                    if opt_attribute_discriminator is not None:
                        opt_attribute_discriminator.zero_grad(set_to_none=False)
                        # Exclude features (last element of batches) for
                        # attribute discriminator
                        with torch.cuda.amp.autocast(
                            enabled=self.config.mixed_precision_training
                        ):
                            generated_output = self._discriminate_attributes(
                                generated_batch[:-1]
                            )
                            real_output = self._discriminate_attributes(real_batch[:-1])

                            loss_generated = torch.mean(generated_output)
                            loss_real = -torch.mean(real_output)
                            loss_gradient_penalty = self._get_gradient_penalty(
                                generated_batch[:-1],
                                real_batch[:-1],
                                self._discriminate_attributes,
                            )

                            attribute_loss = (
                                loss_generated
                                + loss_real
                                + self.config.attribute_gradient_penalty_coef
                                * loss_gradient_penalty
                            )

                        scaler.scale(attribute_loss).backward(retain_graph=True)
                        scaler.step(opt_attribute_discriminator)
                        scaler.update()

                for _ in range(self.config.generator_rounds):
                    opt_generator.zero_grad(set_to_none=False)
                    with torch.cuda.amp.autocast(
                        enabled=self.config.mixed_precision_training
                    ):
                        generated_output = self._discriminate(generated_batch)

                        if self.attribute_discriminator:
                            # Exclude features (last element of batch) before
                            # calling attribute discriminator
                            attribute_generated_output = self._discriminate_attributes(
                                generated_batch[:-1]
                            )

                            loss = -torch.mean(
                                generated_output
                            ) + self.config.attribute_loss_coef * -torch.mean(
                                attribute_generated_output
                            )
                        else:
                            loss = -torch.mean(generated_output)

                    scaler.scale(loss).backward()
                    scaler.step(opt_generator)
                    scaler.update()
