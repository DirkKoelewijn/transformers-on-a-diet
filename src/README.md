# Source code (src)

In this folder, you find all the custom code that is used to parse the data and build, train and evaluate the models:

* In `data.py` you'll find everything to parse and preprocess the data.
* In `components.py` you'll find all custom layers and models that are used as components in our main models.
* In `models.py` you'll find our main models.
* In `callbacks.py` you'll find the callbacks that train the regular and Wasserstein GANS, as well as a callback to add more datasets to evaluate on.
* In `loss.py` you'll find quick functions to calculate the loss for semi-supervised GANs.
* In `experiments.py` you'll find the quick functions to quickly perform an entire experiment.
