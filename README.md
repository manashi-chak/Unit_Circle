# Unit_circle
This project is an implementation of "IRIS VERIFICATION WITH CONVOLUTIONAL NEURAL NETWORK AND UNIT-CIRCLE LAYER". 

Dataloader.py:
1. In this file image is being read from the image list.

2. The loading of images is done in a batchwise manner(not pairwise).Pairwise loading is done in trainer.py file.

3.The label of images are also sent back which is basically a label of the class, that helps in the pairwise loading from trainer.py.




Main.py
1. Main.py is the main calling module.

2. Line 71-173 are the different inputs to be provided from command line.They comprise of the mode of operation( Training or testing),root paths of data files and path where the output will be saved,file names(checkpoint,training data,validation data,testing data,meta files,log files),batch size for training and testing,learning rate, momentum, regularisation decay factor,gpu to use, frequency of saving the checkpoint,frequency of printing status and checkpoint to load(if any).

3.For the first 100 epochs ,the weights are kept fixed and after that the whole network is trained.

The architecture is defined in UC_pairwise_radim.py.

Trainer.py

1. As shown in the figure, a batch of images is dived into two halves and given as input to a shared network. For 2 images given as input to the shared network, we get a single value as output. The target is created by passing these images into a function "classparser" which takes 2 images and check their labels passed on by Dataloader.py.If the labels are same,that is, if they belong to the same class,then the output should be 0.

2. The matcher module is also defined in the class Verifier_pairwise.

3. For each batch consisting of N samples, we have N/2 genuine and N/2 imposter samples.

