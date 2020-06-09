# PolyMusicPredLSTM

This repository contains code for the publication:

> Adrien Ycart and Emmanouil Benetos. "Learning and Evaluation Methodologies for Polyphonic Music Sequence Prediction with LSTMs", _IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)_, 28:1328-1341 (2020).

If you use any of this in your works, please cite:

```  
    @article{ycart2019taslp,
      author    = {Adrien Ycart and
              Emmanouil Benetos},
      title     = {Learning and Evaluation Methodologies for Polyphonic Music Sequence
                    Prediction With LSTMs},
      journal   = {{IEEE} {ACM} Transactions on Audio, Speech, and Language Processing},
      volume    = {28},
      pages     = {1328--1341},
      year      = {2020},
      url       = {https://doi.org/10.1109/TASLP.2020.2987130},
      doi       = {10.1109/TASLP.2020.2987130},
      }
```

## Getting started

This code is compatible with Python 2.7.

To run this code, create a virtual environment, then run: ```$ pip install -r requirements.txt```

## Training a model

To train a model, use the ```script_train.py``` script.
Options can be displayed with  ```$ python script_train.py -h```.

## Evaluating a model

To train a model, use the ```script_eval.py``` script.
Options can be displayed with  ```$ python script_eval.py -h```.
