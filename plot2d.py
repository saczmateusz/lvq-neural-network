from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

x_n = list()
y_p = list()

with open('epoch_10-100_nv_90-100_lr_03/epoch_10-150_nv_90_lr_03_2_ep.txt', 'r') as xf:
    for x in xf:
        x_n.append(int(x))

with open('epoch_10-100_nv_90-100_lr_03/epoch_10-150_nv_90_lr_03_3_pk.txt', 'r') as yf:
    for y in yf:
        y_p.append(float(y))

plt.plot(x_n, y_p, 'r')
plt.xlabel('Epoki')
plt.ylabel('PK [%]', rotation='vertical')
plt.axis([10, 150, 90, 100])
plt.grid(True)
plt.show()
