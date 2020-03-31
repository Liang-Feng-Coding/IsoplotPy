import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
from numpy import shape
#from sklearn.model_selection import cross_val_predict
import csv

filename = "ellipse error - python.csv"

with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)

    for index, column_header in enumerate(header_row):
        print(index, column_header)

    xnum = []
    ynum = []
    x_err_num = []
    y_err_num = []
    
    for row in reader:
        x = float(row[0])
        xnum.append(x)

        y = float(row[2])
        ynum.append(y)

        x_err = float(row[1])
        x_err_num.append(x_err)

        y_err = float(row[3])
        y_err_num.append(y_err)

x = np.array(xnum)
y = np.array(ynum)
x1 = x.reshape(-1,1)
y1 = y.reshape(-1,1)
x_err = np.array(x_err_num)
y_err = np.array(y_err_num)

model = LinearRegression()
model.fit(x1, y1)
y_pred = model.predict(x1)

#predicted = cross_val_predict(model, x1, y1, cv=10)

fig, ax = plt.subplots()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
ells = [Ellipse((float(x[i]), float(y[i])), float(x_err[i])*2.4, float(y_err[i])*2.4, linewidth=0.5, edgecolor='black', facecolor='none') for i in range(len(x))]

a = plt.subplot(111)

for e in ells:
    #e.set_clip_box(a.bbox)
    #e.set_alpha(0.9)
    a.add_artist(e)
    
plt.plot(x1, y_pred, color='r', linewidth=0.2)

plt.xlabel(r'$\mathregular{^{238}U/^{206}Pb}$')
plt.ylabel(r'$\mathregular{^{207}Pb/^{206}Pb}$')

'''
xmin,xmax,ymin,ymax = input("输入x轴最小值-最大值-y轴最小值-最大值，以逗号隔开：").split(',')
xmin = float(xmin)
xmax = float(xmax)
ymin = float(ymin)
ymax = float(ymax)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
'''
plt.xlim(0, max(x)+5)
plt.ylim(0, max(y)+1)

plt.show()
