import quadvar as qv
import quadmodel as qm
import numpy as np
from array import *

# run from 0 to t_plot
while qv.t_plot[qv.counter-1] < max(qv.t_plot):
    exec(open("quadmodel.py").read());

# plot the result
exec(open("quadplot.py").read());