### Scatter matrix plot
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# np.random.seed(134)                     
# N = 1000                              
 
# x1 = np.random.normal(0, 1, N)                        
# x2 = x1 + np.random.normal(0, 3, N)              
# x3 = 2 * x1 - x2 + np.random.normal(0, 2, N)

# df = pd.DataFrame({'x1':x1,
#                    'x2':x2,
#                    'x3':x3})

# pd.plotting.scatter_matrix(df)
# plt.show(block=True)

### Box plots
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Polygon


# # Fixing random state for reproducibility
# np.random.seed(19680801)

# # fake up some data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 50
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# data = np.concatenate((spread, center, flier_high, flier_low))

# fig, axs = plt.subplots(2, 3)

# # basic plot
# axs[0, 0].boxplot(data)
# axs[0, 0].set_title('basic plot')

# # notched plot
# axs[0, 1].boxplot(data, 1)
# axs[0, 1].set_title('notched plot')

# # change outlier point symbols
# axs[0, 2].boxplot(data, 0, 'gD')
# axs[0, 2].set_title('change outlier\npoint symbols')

# # don't show outlier points
# axs[1, 0].boxplot(data, 0, '')
# axs[1, 0].set_title("don't show\noutlier points")

# # horizontal boxes
# axs[1, 1].boxplot(data, 0, 'rs', 0)
# axs[1, 1].set_title('horizontal boxes')

# # change whisker length
# axs[1, 2].boxplot(data, 0, 'rs', 0, 0.75)
# axs[1, 2].set_title('change whisker length')

# fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
#                     hspace=0.4, wspace=0.3)

# # fake up some more data
# spread = np.random.rand(50) * 100
# center = np.ones(25) * 40
# flier_high = np.random.rand(10) * 100 + 100
# flier_low = np.random.rand(10) * -100
# d2 = np.concatenate((spread, center, flier_high, flier_low))
# # Making a 2-D array only works if all the columns are the
# # same length.  If they are not, then use a list instead.
# # This is actually more efficient because boxplot converts
# # a 2-D array into a list of vectors internally anyway.
# data = [data, d2, d2[::2]]

# # Multiple box plots on one Axes
# fig, ax = plt.subplots()
# ax.boxplot(data)

# plt.show()

### Histogram
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Create a random number generator with a fixed seed for reproducibility
rng = np.random.default_rng(19680861)

N_points = 100000
n_bins = 20

# Generate two normal distributions
dist1 = rng.standard_normal(N_points)
dist2 = 0.4 * rng.standard_normal(N_points) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(dist1, bins=n_bins)
axs[1].hist(dist2, bins=n_bins)

plt.show()