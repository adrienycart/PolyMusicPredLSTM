# PolyMusicPredLSTM

This repository contains code for the publication:

> Adrien Ycart and Emmanouil Benetos. "Learning and Evaluation Methodologies for Polyphonic Music Sequence Prediction with LSTMs", _IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)_, Under Review.

If you use any of this in your works, please cite:

```  
    @article{ycart2019taslp,
       Author = {Ycart, Adrien and Benetos, Emmanouil},    
       Booktitle = {IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)},    
       Title = {Learning and Evaluation Methodologies for Polyphonic Music Sequence Prediction with LSTMs},    
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
