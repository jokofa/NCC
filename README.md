# Neural Capacitated Clustering

---
This is the repo accompanying our paper 
"Neural Capacitated Clustering"
accepted at the 32nd International Joint Conference on Artificial Intelligence
(IJCAI23).

The preprint version of our paper can be found [here](https://arxiv.org/abs/2302.05134)

---


### setup

install via conda:
````
conda env create -f requirements.yml
````
the data is stored via [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

### Evaluation
Simply run the corresponding notebooks to prepare the data and run the evaluation.


### Training

In order to create training data, first sample data via [ccp_create_data.ipynb](ccp_create_data.ipynb).
Then run the training task runner via [run.py](run.py). The model is configured via hydra. 
The configuration files can be found in the config directory. 
The default config can also be shown via the -h flag
````
python run.py -h
````
additional arguments can simply be provided via the command line.