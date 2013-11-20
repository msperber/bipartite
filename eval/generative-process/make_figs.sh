#!/bin/bash

gamma=2
tau=1


alpha=1
sigma=0
../../bipartite-py/generate.py -g $gamma -a $alpha -t $tau -s $sigma -n 30 | ../../bipartite-py/plot_bipartite.py -o fig.alpha=$alpha.sigma=$sigma.png

alpha=5
sigma=0
../../bipartite-py/generate.py -g $gamma -a $alpha -t $tau -s $sigma -n 30 | ../../bipartite-py/plot_bipartite.py -o fig.alpha=$alpha.sigma=$sigma.png

alpha=10
sigma=0
../../bipartite-py/generate.py -g $gamma -a $alpha -t $tau -s $sigma -n 30 | ../../bipartite-py/plot_bipartite.py -o fig.alpha=$alpha.sigma=$sigma.png

alpha=2
sigma=0.1
../../bipartite-py/generate.py -g $gamma -a $alpha -t $tau -s $sigma -n 30 | ../../bipartite-py/plot_bipartite.py -o fig.alpha=$alpha.sigma=$sigma.png

alpha=2
sigma=0.5
../../bipartite-py/generate.py -g $gamma -a $alpha -t $tau -s $sigma -n 30 | ../../bipartite-py/plot_bipartite.py -o fig.alpha=$alpha.sigma=$sigma.png

alpha=2
sigma=0.9
../../bipartite-py/generate.py -g $gamma -a $alpha -t $tau -s $sigma -n 30 | ../../bipartite-py/plot_bipartite.py -o fig.alpha=$alpha.sigma=$sigma.png

