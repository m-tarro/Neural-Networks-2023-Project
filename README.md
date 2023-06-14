# Neural Networks 2023 Project

For us humans it is often difficult to grasp how vastly diverse the natural systems around us really are. However, understanding this extreme diversity is essential for addressing the challenges humanity faces. Identifying the extraordinary range of species is a daunting task and flowers are no exception in this case, with approximately 400,000 species alone. Thus botanical studies and different conservation programmes would greatly benefit from some sort of automation mechanism in identifying them. 

Machine learning models could potentially offer a promising solution, using computer vision and pattern recognition techniques to analyse flower images and quickly provide accurate identification. Unfortunately flower identification with machine learning models is a challenging task due to the vast diversity of floral species, their complex morphological features, and variations in colour, shape, and texture. Identifying a particular species of flower often requires expertise and a detailed understanding of their characteristics and preferred habitats, which is not always easy to capture in an image.

The goal of this project is to explore which augmentation techniques work the best on the given flower images and see if some augmentation techniques work better or worse on different model architectures.

The summary of the project can be found on [Medium](https://medium.com/@marcus.artner/petals-to-metal-kaggle-competition-b67f6bf84e2b).

Model selection was done in [flower-classification-notebook.ipynb](/flower-classification-notebook.ipynb). The best model that was run on test data can be found in [flower-classification-notebook-manual-augmentation-densenet.ipynb](/flower-classification-notebook-manual-augmentation-densenet.ipynb). The notebooks have to be run in Kaggle on TPU VM. The dependencies are installed in the first cells.
