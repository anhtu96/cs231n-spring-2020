# CS231n
My coding solutions to Stanford's CS231n, Spring 2020.

This repository contains 3 assignments as below:
### [Assignment #1](https://cs231n.github.io/assignments2020/assignment1/): Image Classification, kNN, SVM, Softmax, Fully-Connected Neural Network 
- Done all.
- Using local development.
### [Assignment #2](https://cs231n.github.io/assignments2020/assignment2/): Fully-Connected Nets, BatchNorm, Dropout, ConvNets, Tensorflow/Pytorch 
- Done all, except the task *Batch normalization: alternative backward* in [BatchNormalization.ipynb](/assignment2/BatchNormalization.ipynb). For this task, you might want to refer to this [great article](http://cthorey.github.io./backpropagation/).
- Done TensorFlow tasks only, PyTorch notebooks are not touched.
### [Assignment #3](https://cs231n.github.io/assignments2020/assignment3/): Image Captioning with Vanilla RNNs and LSTMs, Neural Net Visualization, Style Transfer, Generative Adversarial Networks
- Done all (PyTorch notebooks are not touched).

### Important notes
1. For **Assignment #1**, please for follow the instructions on the course's page on how to setup virtual environment, packages, etc.
2. For **Assignment #2** and **#3**, I use Google Colab to run all the codes. The course's page also provides instructions on how to get things prepared. If you want to use Google Colab, PLEASE change the ```FOLDERNAME``` value at the beginning code cell of each notebook, depending on your working directory. 
3. In **Assignment #3**, for the notebook [StyleTransfer-TensorFlow.ipynb](/assignment3/StyleTransfer-TensorFlow.ipynb), the errors for content loss, style loss, etc. are high. I think my implementation is correct. I did some investigation and found that the values from the file [style-transfer-checks-tf.npz](/assignment3/style-transfer-checks-tf.npz) are strange. For example, I compared the values of ```cl_out``` key (content loss) of files used in 2019 and 2020 course. The results are as follow:
    * ```style-transfer-checks-tf.npz``` (TensorFlow) of 2020: ```cl_out = 60549.98```. 
    * ```style-transfer-checks.npz``` (PyTorch) of 2020: ```cl_out = 32436.986```.
    * ```style-transfer-checks-tf.npz``` (TensorFlow) of 2019: ```cl_out = 32448.51```.
    * ```style-transfer-checks.npz``` (PyTorch) of 2019: ```cl_out = 32436.986```.

    You can see that the difference between the 2020's TF value and the others is HUGE. If you have any ideas about it, feel free to add comments on my commit, or contact me via email <tungo.ee.96@gmail.com>.