# Denoising Autoencoder Energy Disaggregator

As described in:

[Neural NILM: Deep Neural Networks Applied to Energy Disaggregation](https://arxiv.org/pdf/1507.06594.pdf) by Jack Kelly and William Knottenbelt

See example experiment [here](https://github.com/OdysseasKr/neural-disaggregator/blob/master/DAE/DAE-example.ipynb).

DAE-example.ipynb - code test for notebooks

daedisaggregator.py - Tensorflow backend (NN based on keras utilities)

daedisaggregator_pytorch.py - (NN based on Pytorch utilities) --> Still DAE, not VAE!!!

daedisaggregator_pytorch4.py - (Variational Autoencoder based on Pytorch utilities) --> UPDATED version

disag-out.h5 - h5 file for the resulting datastore

metrics.py - to serve model evaluation

model-redd100.h5 - the file where the trained model is saved for later use

redd-test.py - run the disaggregation code

ukdale-test.py - run the dissagregation code for UKDALE dataset

nilmtk-env-clone.yml - type: "conda env create -f nilmtk-env-clone.yml" in order to set the environment
