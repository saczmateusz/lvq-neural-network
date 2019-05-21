from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.mlab import bivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import array

x_axis = set()
y_axis = set()
z_val = list()

with open('lr_test_001-100_nv_test_10-100_ep_100/lr_test_001-100_nv_test_10-100_ep_100_1_lr.txt', 'r') as xf:
    for x in xf:
        x_axis.add(float(x))

with open('lr_test_001-100_nv_test_10-100_ep_100/lr_test_001-100_nv_test_10-100_ep_100_2_cb.txt', 'r') as yf:
    for y in yf:
        y_axis.add(int(y))

with open('lr_test_001-100_nv_test_10-100_ep_100/lr_test_001-100_nv_test_10-100_ep_100_3_pk.txt', 'r') as zf:
    for z in zf:
        z_val.append(float(z))

x_axis = list(x_axis)
x_axis.sort()
lx = len(x_axis)

y_axis = list(y_axis)
y_axis.sort()
ly = len(y_axis)

Z_matrix = []
for i in range(lx):
    row = []
    for j in range(ly):
        row.append(float(z_val[j*lx + i]))
    Z_matrix.append(row)

Z_matrix = np.array(Z_matrix)

X, Y = np.meshgrid(x_axis, y_axis)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z_matrix, rstride=1, cstride=1, cmap=plt.cm.jet)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('LR')
ax.set_ylabel('S')
ax.set_zlabel('PK [%]', rotation='vertical', labelpad=7)

fig.colorbar(surf, shrink=1, aspect=20)

plt.show()

xf.close()
yf.close()
zf.close()
