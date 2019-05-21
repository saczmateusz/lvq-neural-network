from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt

x_n = list()
y_p = list()

with open('ep.txt', 'r') as xf:
    for x in xf:
        x_n.append(int(x))

with open('pk.txt', 'r') as yf:
    for y in yf:
        y_p.append(float(y))

plt.plot(x_n, y_p, 'r')
plt.xlabel('Epoki')
plt.ylabel('PK [%]', rotation='vertical')
plt.axis([10, 200, 90, 100])
plt.grid(True)
plt.show()
