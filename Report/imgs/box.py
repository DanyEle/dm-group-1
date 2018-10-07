#!/usr/bin/python3
from scipy.stats import spearmanr, kendalltau
import sys

lines = [list(map(float, l.split(','))) for l in open(sys.argv[1])]

i = int(sys.argv[2])
j = int(sys.argv[3])
print(lines[i], lines[j])
