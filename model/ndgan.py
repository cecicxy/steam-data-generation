# Create some random training data
import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig
from gretel_synthetics.timeseries_dgan.config import OutputType
import torch
import pandas as pd
torch.manual_seed(133)
np.random.seed(1) 
# Logs = []
# def callback(ProgressInfo):
#     log = ProgressInfo.loss
#     log['epoch'] = ProgressInfo.epoch
#     print("epoch: {}--- loss vary:{}".format(log['epoch'],log['loss']))
#     Logs.append(log)
if __name__ == "__main__":
    attributes = np.load("newdata/attributes.npy")
    features = np.load("newdata/features.npy")
    
    print(features.shape)

    model = DGAN(DGANConfig(

    max_sequence_len=features.shape[1],

    sample_len= 4, #lstm一个cell生成的点数，必须整除144
    attribute_noise_dim = 6,  #几个noice生成label
    feature_noise_dim= 48,  #需要调参
    # generator_rounds= 2,
    # discriminator_rounds=3,
    batch_size=512,

    epochs=6959,  # For real data sets, 100-1000 epochs is typical

    ))
    model.train_numpy(

        features= features,
        attributes = attributes,
        attribute_types = [OutputType.DISCRETE] *attributes.shape[-1] ,
        # progress_callback= callback

        )
    model.save("newdata/model/mode100000")
    gen_attributes, gen_features = model.generate_numpy(features.shape[0])
    np.save("./newdata/result/gen_sample_attributes",gen_attributes)
    np.save("./newdata/result/gen_sample_features",gen_features)
    # pd.DataFrame(Logs).to_csv("newdata/result/logs.csv")