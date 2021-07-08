from __future__ import print_function, division
import random
import sys

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py

# Pytorch packages
import torch
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

#from keras.models import load_model
#from keras.models import Sequential
#from keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
#from keras.utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.legacy.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore


class AE(nn.Module):

    def __init__(self,sequence_len):
        #sequence_len=256
        super().__init__()
        self.sequence_len = 256
        self.encoder=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, padding='same',stride = 1),
            nn.Flatten(),
            nn.Dropout(0.2),
            #nn.Linear((sequence_len - 0) * 8, (sequence_len - 0) * 8),
            nn.Linear(sequence_len*8, sequence_len*8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(sequence_len*8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.decoder=nn.Sequential(
            #nn.Linear(128, (sequence_len - 0) * 8),
            #nn.ReLU(),
            #nn.Dropout(0.2),
            #nn.Unflatten(((sequence_len - 0)*8), ((sequence_len - 0),8)),
            #nn.ConvTranspose1d(out_channels=1, kernel_size=4, in_channels=sequence_len, padding='same', stride=1)
            nn.Linear(128, sequence_len*8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Unflatten(1, (8,256)),
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=4, padding=131, stride=2, output_padding=1, dilation=2)
            #nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=4)
        )

    def forward(self, x):
        #encoded=self.encoder(x)
        #decoded=self.decoder(encoded)
        #x = x.permute(0,2,1)
        for layer in self.encoder:
            x = layer(x)
            print(x.size())
        print("decoding...")
        for layer in self.decoder:
            x = layer(x)
            print(x.size())
        self.decoded = x

        # Add one zero at the end
        #self.decoded.expand(-1,-1, 256)

        return self.decoded


class DAEDisaggregator(Disaggregator):
    '''Denoising Autoencoder disaggregator from Neural NILM
    https://arxiv.org/pdf/1507.06594.pdf

    Attributes
    ----------
    model : keras Sequential model
    sequence_length : the size of window to use on the aggregate data
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, sequence_length):
        '''Initialize disaggregator

        Parameters
        ----------
        sequence_length : the size of window to use on the aggregate data
        meter : a nilmtk.ElecMeter meter of the appliance to be disaggregated
        '''
        self.MODEL_NAME = "AUTOENCODER"
        self.mmax = None
        self.sequence_length = sequence_length
        self.MIN_CHUNK_LENGTH = sequence_length
        self.model = self._create_model(self.sequence_length)

    def train(self, mains, meter, epochs=1, batch_size=16, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.ElecMeter object for the meter data
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)

        # Train chunks
        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        if self.mmax == None:
            self.mmax = mainchunk.max()

        while(run):
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)

            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
            except:
                run = False

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        '''

        s = self.sequence_length
        #up_limit =  min(len(mainchunk), len(meterchunk))
        #down_limit =  max(len(mainchunk), len(meterchunk))

        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        # Create array of batches
        #additional = s - ((up_limit-down_limit) % s)
        additional = s - (len(ix) % s)
        X_batch = np.append(mainchunk, np.zeros(additional))
        Y_batch = np.append(meterchunk, np.zeros(additional))

        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s, 1))  # apo edw allages
        Y_batch = np.reshape(Y_batch, (int(len(Y_batch) / s), s, 1))
        print("testt")
        print(type(X_batch[0][0][0]))
        print(X_batch.shape)
        X_batch = np.transpose(X_batch, (0, 2, 1))
        Y_batch = np.transpose(Y_batch, (0, 2, 1))

        #self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

        # prepare pytorch dataloader
        dataset = TensorDataset(torch.tensor(X_batch, dtype=torch.float32), torch.tensor(Y_batch, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # batch size na to megaloso px 1024

        # Training loop
        for epoch in range(2):   # prepei na tautizetai me to epochs sto dae.train(...,epochs=__,...) ???????
            for x, y in dataloader:
                optimizer.zero_grad()

                # forward and backward pass
                print(x.size())
                print("pame!!")
                out = self.model(x)
                '''
                out2 = torch.zeros(16, 1, 256)

                out2[:, :, :255] = out
                print("epitelous")
                print(out2.size())


                #new_array = np.zeros((16, 1, 256))
                #new_array[:, :, :255] = out
                #out = new_array
 
                #out.expand(16, 1, 256)
                '''
                #y2=y[:,:,:255]
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                print(loss.item())  # loss should be decreasing
                print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')

    def _create_model(self, sequence_len):
        model = AE(sequence_len)
        return model

    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        meter_metadata : metadata for the produced output
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.mmax)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series to disaggregate
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        s = self.sequence_length
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        additional = s - (up_limit % s)
        X_batch = np.append(mains, np.zeros(additional))
        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s ,1))

        #pred = self.model.predict(X_batch)
        print("Prin tin allagi...")
        print(X_batch.shape)
        X_batch = np.transpose(X_batch, (0, 2, 1))
        print("Meta tin allagi...")
        print(X_batch.shape)


        X_batch = torch.tensor(X_batch, dtype=torch.float32)

        self.model.eval()  #with torch.no_grad(): na to dokimasw pws tha duleue me to with kai ola ta alla mesa
        pred = self.model(X_batch)

        print(pred)
        pred=pred.detach().numpy()
        print(pred)
        pred = np.reshape(pred, (up_limit + additional))[:up_limit] ##
        column = pd.Series(pred, index=mains.index, name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers


    def import_model(self, filename):
        '''Loads keras model from h5

        Parameters
        ----------
        filename : filename for .h5 file

        Returns: Keras model
        '''
        #self.model = load_model(filename)

        #self.model = torch.load(filename)
        self.model.load_state_dict(torch.load(filename))   ##
        self.model.eval()  ## axreiasto!
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('mmax')
            self.mmax = np.array(ds)[0]

    def export_model(self, filename):
        '''Saves keras model to h5

        Parameters
        ----------
        filename : filename for .h5 file
        '''
        #self.model.save(filename)
        torch.save(self.model.state_dict(), filename)
        print("egine to save")
        with h5py.File(filename, 'w') as hf:           ##### allaksa to a me w
            print("no prob 1")
            gr = hf.create_group('disaggregator-data')
            print("no prob 2")
            gr.create_dataset('mmax', data = [self.mmax])
            print("egine kai to allo")

