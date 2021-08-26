ADNet implementation in Python using PyTorch and [PyTracking](https://github.com/visionml/pytracking).


# About

The tracker is trained using supervised learning and a modified REINFORCE policy gradient algorithm.

We show that using a curriculum speeds up reinforcement learning.
The curriculum is built from synthetic sequences gradually increasing in difficulty.

This repository is part of my undergraduate thesis.


# Demo

Tracking on a synthetic sequence, without fine tuning, trained using only reinforcement learning.

![Duck demo](demos/motocross1_duck.gif)

<!-- ![Fish demo](demos/fish1_fish2.gif) -->



# Setup

The tracker is incorporated into the (modified) PyTracking library.  
* Tracker evaluation and training source code files are found in [pytracking/pytracking/tracker/adnet/](pytracking/pytracking/tracker/adnet/).  
* Tracker parameter files are found in [pytracking/pytracking/parameter/adnet/](pytracking/pytracking/parameter/adnet/default.py).  
* Included is a simple [synthetic sequence generator](pytracking/pytracking/tracker/adnet/synthetic.py).

For setup instructions and a training demonstration see the example Jupyter [notebook](demos/demo.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mare5x/adnet-rl-vot/blob/master/demos/demo.ipynb)


