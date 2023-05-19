# Neural Capacitated Clustering

---
This is the repo accompanying our paper 
"Neural Capacitated Clustering"
accepted at the 32nd International Joint Conference on Artificial Intelligence
(IJCAI23).

The preprint version of our paper can be found [here](https://arxiv.org/abs/2302.05134).

---


### Setup

install via conda:
````
conda env create -f requirements.yml
````
the data.zip is stored via [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).
You have to download and unpack it before you can use it.


---
### Evaluation
Simply run the corresponding notebooks to prepare the data and run the evaluation.



---
### Training

In order to create training data, first sample data via [ccp_create_data.ipynb](ccp_create_data.ipynb).
Then create labels via a method of choice using 
[ccp_create_labels.py](ccp_create_labels.py) or [vrp_create_labels.py](vrp_create_labels.py).
Finally run the training task runner via [run.py](run.py). The model is configured via hydra. 
The configuration files can be found in the config directory. 
The default config can also be shown via the -h flag
````
python run.py -h
````
additional arguments can simply be provided via the command line.

