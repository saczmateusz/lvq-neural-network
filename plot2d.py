from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

x_n = list()
y_p = list()

with open('neurons10-300_lr_03_ep_100/neuron_test_10-200_2_cb.txt', 'r') as xf:
    for x in xf:
        x_n.append(int(x))

with open('neurons10-300_lr_03_ep_100/neuron_test_10-200_3_pk.txt', 'r') as yf:
    for y in yf:
        y_p.append(float(y))

plt.plot(x_n, y_p, 'r')
plt.xlabel('Liczba neuronów')
plt.ylabel('Poprawność klasyfikacji [%]', rotation='vertical')
plt.axis([10, 300, 90, 100])
plt.grid(True)
plt.show()
