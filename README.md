# Projet RI - 
## Usage
The models are deployed in the notebook called "Final_notebook". To execute it, one just need to follow the guidelines indicated in the notebook. 

## Pipeline 
We implemented several models : boolean search, vectorial model, language model. 
Indexing, data preparation is included in the notebook. The vectorial and the language models are deployed in two separate scripts, namely vectorial_model.py and language_model.py. 

## Results
When taking 11 recall values to compute the mean average precision, we obtain the following results for our models : 

| ﻿                                                  | Mean average precision, K=11 |
|---------------------------------------------------|------------------------------|
| Modèles vectoriel : Fréquence                     | 0.568                        |
| Modèles vectoriel : TF IDF normalisé              | 0.557                        |
| Modèles vectoriel : TF IDF logarithmique          | 0.313                        |
| Modèles vectoriel :TF IDF normalisé logarithmique | 0.582                        |
| Modèles vectoriel : binaire                       | 0.551                        |
| Modèle de langue                                  | 0.165                        |

So the vectorial model with TD IDF normalized and logarithmic seems to be the most appropriate for our set of documents. 
