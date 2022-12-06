# Transformers on a Diet

This is the GitHub repository of for my master thesis: [Transformers on a Diet: Semi-Supervised Generative Adversarial Networks for Reducing Transformers' Hunger for Data in Aspect-Based Sentiment Analysis](thesis.pdf). It contains all source code to build, train and evaluate the models in my thesis and the detailed results for all experiments.


## Structure

This repo has the follo[](https://)wing structure:

* In `data` you'll find the datasets.
* In `examples` you'll find Jupyter notebooks showing how to parse data and build, train and evaluate the models for our experiments.
* In `img` you'll find high-res versions of the images used in the thesis.
* In `results` you'll find the detailed results.
* In `src` you'll find the source code for all this.

Each sub-folder contains a Readme file that can help you further.

## Licence

This repo uses the [MIT License](LICENSE.md).

## Citing

If you wish to cite this repo or my thesis, please use:

```plaintext
@misc{koelewijn2022,
            year = {2022},
          author = {Dirk {Koelewijn}},
           title = {Transformers on a Diet: Semi-Supervised Generative Adversarial Networks for Reducing Transformers' Hunger for Data in Aspect-Based Sentiment Analysis},
           month = {December},
             url = {http://essay.utwente.nl/93794/},
        abstract = {The vast amount of reviews and opinions being shared online for practically all available goods and services has an enormous potential value. Although the current state-of-the-art Aspect-Based Sentiment Analysis (ABSA) methods show impressive results in extracting valuable opinions, these Transformer-based models require large high-quality annotated training datasets. Datasets that are not always available and which are very costly to create. To reduce the hunger of these models for annotated data, we for the first time apply Generative Adversarial Networks (GANs) to ABSA. We investigate using both regular and Wasserstein semi-supervised GANs to generate artificial word embeddings, with varying amounts of unlabelled data and varying generator complexity. We show that adding such a GAN can significantly improve performance, even without using unlabelled data. Furthermore, we identify how much unlabelled data works best and show that generators with more hidden layers perform better. Altogether, we show that our method allows for reducing annotated data by 50\% while still achieving similar performance.}
}
```
