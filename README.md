# FFA_test
- Code modified from **mohammadpz/pytorch_forward_forward**.
- Added implementation on CIFAR-10, with additional weight-decay and dropout layer.
- Incorporated wandb sweep functionality into original codes by using config files.
- Modified training pipeline: training one layer for 1000 epochs, each epoch containing 10 minibatches, then progressing into the following layer (and will not return to the previous layer later on)
- Published a branch called **"local_receptive"**, where a technique of local receptive field was implemented in order to reduce overfitting (according to Hinton's FFA13 paper Chapter 4)
- Published a branch called **output_layer**, where the implementation of full_linear_net.py and linear_layer.py were **reorganized** into multiple classes, and a **fully-connected output layer** was also added to the previous local_receptive net.