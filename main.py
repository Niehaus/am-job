# -*- coding: utf-8 -*-
"""
@author: amm.bernardes@gmail.com
@author: barbs.boechat@gmail.com

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import time
import os
import csv
from pathlib import Path

if __name__ == '__main__':
	pass