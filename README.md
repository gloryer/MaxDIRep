
# Maximizing Information in Domain-Invariant Representation Improves Transfer Learning


## Requirements

 - TensorFlow=2.9.2 
 - Python 3.9


## Network Architecture


<img src="fig/VAEGAN_page.jpg" width="550"/>



## Running the Experiment

This repository implements the experiments mentioned in the paper.

**CIFAR-10 with cheating-color-plane** 
    python cifar_bias.py 

**Fashion-MNIST with cheating-bit** 
    python fm_nocheat.py
    python fm_random.py
    python fm_shift.py
**Digits datasets**

MINIST to MNIST-M, SVHN to MNIST, Synth Digits to SVHN
    python mnist_mnistm.py
    python svhn_mnist.py
    python synth_svhn.py

**Office dataset**

DSLR to Amazon, Webcam to Amazon
    python office_D_A.py
    python office_W_A.py

GitHub limits the size of files allowed in repositories so we omit the datasets, which will be released publicly for study. 

[//]: # ()
[//]: # (## Results)

[//]: # (Fashion-MNIST with cheating-bit)

[//]: # (|  Model | No <br>cheating | Shift <br>cheating | Random <br>cheating |)

[//]: # (|--------|-------------|----------------|-----------------|)

[//]: # (| VAEGAN |        66.8 |           65.7 |            61.6 |)

[//]: # ()
[//]: # (CIFAR-10 with cheating-color-plane)

[//]: # (|  Model | 0% <br>bias | 20% <br>bias | 40% <br>bias | 60% <br>bias | 80% <br>bias | 90% <br>bias | 100% <br>bias |)

[//]: # (|--------|---------|----------|----------|----------|----------|----------|-----------|)

[//]: # (| VAEGAN |    70.4 |     69.8 |     69.8 | 69.7     | 68.3     | 64.1     | 34.2      |)

[//]: # ()
[//]: # (Digits datasets )

[//]: # (|  Model | MNINST to <br>MNIST-M | Synth Digits to <br>SVHN | SVHN to <br>MNIST |)

[//]: # (|--------|-------------------|----------------------|---------------|)

[//]: # (| VAEGAN |              81.0 |                 91.1 |          85.8 |)



### Reconstructed Images


<img src="fig/reconstruction-FM.jpg" width="500"/>

 Columns 1 and 4, original images; 2 and 6, reconstructions
of originals; 3 and 5, reconstructions with domain bit flipped.