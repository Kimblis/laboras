import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sympy import*

# Apibrezimo sritis
xrang = np.array([-5, 5])
yrang = np.array([-2, 2])
s = 5


def grafikas(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, alpha=0.6)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()


def grid(xrang, yrang, step):
    xstep = (xrang[1]-xrang[0])/step
    ystep = (yrang[1]-yrang[0])/step
    x = xrang[0]
    cords = []
    for i in range(0, step+1):
        y = yrang[0]
        for j in range(0, step+1):
            cords.append(np.array([x, y]))
            y += ystep
        x += xstep
    return cords


def gradientas(t, Fdx, Fdy):
    grad = np.array([Fdx.subs([(x, t[0]), (y, t[1])]),
                     Fdy.subs([(x, t[0]), (y, t[1])])])
    grad = np.float64(grad)
    grad = - grad/ilgis(grad[0], grad[1])*alpha
    return grad


def ilgis(x, y):
    return np.sqrt(x**2 + y**2)


# Grafiko braizymui
xx = np.arange(-5, 6, 1)
yy = np.arange(-2, 3, 1)
xx, yy = np.meshgrid(xx, yy)
zz = -(yy - np.cos(xx+2*yy) + np.sin(xx-3*yy))
grafikas(xx, yy, zz)

# Funkcija ir jos dalines isvestines
F, x, y, Fdx, Fdy = symbols('F x y Fdx Fdy')
F = (y - cos(x+2*y) + sin(x-3*y))
Fdx = diff(F, x)
Fdy = diff(F, y)

gridas = grid(xrang, yrang, s)

opt = true
t = gridas[0]
alpha = 0.1
eps = 1e-5
fnk2 = -1e10
fnk = F.subs([(x, t[0]), (y, t[1])])
iterNr = 1
iterMax = 700
stp_dir = 0.5
grad = gradientas(t, Fdx, Fdy)
t2 = t + grad


for i in gridas:
    t = i
    grad = gradientas(t, Fdx, Fdy)
    print(grad)
    while (abs(grad[0]) > eps or abs(grad[1]) > eps) and (ilgis(t[0]-t2[0], t[1]-t2[1]) > eps) and (iterNr < iterMax):
        if(fnk > fnk2) or opt:
            grad = gradientas(t, Fdx, Fdy)
            grad = grad/alpha
            if fnk > fnk2:
                alpha = alpha*stp_dir
        fnk2 = fnk
        fnk = F.subs([(x, t[0]+grad[0]*alpha),
                      (y, t[1]+grad[1]*alpha)])
        iterNr += 1
        t2 = t
        t = t + grad * alpha
        print("Iteracijoje ", iterNr, "zingsnis ", alpha, "o artinys", t)
        print(fnk)
    print('Minimumo tasko artinys: ', t, 'po ', iterNr, 'iteraciju')
