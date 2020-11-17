import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sympy import*
from mpl_toolkits.mplot3d import Axes3D

# Apibrezimo sritis, bei 6x5 tinklelis(gridas)
xrang = np.array([-5, 5])
yrang = np.array([-2, 2])
a = 6
b = 5


def f(t):
    return F.subs([(x, t[0]), (y, t[1])])


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


def grid(xrang, yrang, a, b):
    xstep = (xrang[1]-xrang[0])/(a-1)
    ystep = (yrang[1]-yrang[0])/(b-1)
    x = xrang[0]
    cords = []
    for i in range(0, a):
        y = yrang[0]
        for j in range(0, b):
            cords.append(np.array([x, y]))
            y += ystep
        x += xstep
    return cords


def uniq(items):
    u = []
    u.append(items[0])
    for i in range(1, len(items)):
        for j in range(len(u)):
            if np.all(items[i] == u[j]):
                break
            else:
                u = u + [items[i]]
    return u


def gradientas(t, Fdx, Fdy):
    grad = np.array([Fdx.subs([(x, t[0]), (y, t[1])]),
                     Fdy.subs([(x, t[0]), (y, t[1])])])
    grad = np.float64(grad)
    grad = - grad/ilgis(grad[0], grad[1])*alpha
    return grad


def ilgis(x, y):
    return np.sqrt(x**2 + y**2)


# Grafiko braizymui
xx = np.arange(-5, 5, 0.1)
yy = np.arange(-2, 2, 0.1)
xx, yy = np.meshgrid(xx, yy)
zz = (yy - np.cos(xx+2*yy) + np.sin(xx-3*yy))
grafikas(xx, yy, zz)

# Funkcija ir jos dalines isvestines
F, x, y, Fdx, Fdy = symbols('F x y Fdx Fdy')
F = (y - cos(x+2*y) + sin(x-3*y))
Fdx = diff(F, x)
Fdy = diff(F, y)

alpha = 0.1
gridas = grid(xrang, yrang, a, b)
ekstremumai = []
for i in gridas:
    t = i
    iterNr = 1
    minimalus_skirtumas = 0.2
    alpha = 0.1
    fnk2 = -1e10
    opt = true
    eps = 1e-5
    iterMax = 700
    stp_dir = 0.5
    fnk = f(t)
    grad = gradientas(t, Fdx, Fdy)
    t2 = t + grad
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
    if len(ekstremumai) == 0:
        ekstremumai.append(t)
    else:
        tinka = False
        for eks in ekstremumai:
            p1 = (eks[0] - t[0])**2
            p2 = (eks[1] - t[1])**2
            skirtumas = sqrt(p1+p2)
            if skirtumas > minimalus_skirtumas:
                tinka = True
                break
        if tinka:
            ekstremumai.append(t)
    #     print("Iteracijoje ", iterNr, "zingsnis ", alpha, "o artinys", t)
    # print('Minimumo tasko artinys: ', t, 'po ', iterNr, 'iteraciju')


# print(ekstremumai)
# minimumai = []
# for grad in ekstremumai:
#     minimumas = f(grad)
#     minimumai.append(minimumas)

# print(minimumai)
# gradmin = None
# z = 0
# for grad in ekstremumai:
#     if f(grad) > z:
#         z = f(grad)
#         gradmin = grad

# print(f'Gautas minimumo taskas:( {str(gradmin[0])} , {str(gradmin[1])} )')
# print(f'Gautas minimumas: {str(z)}')

# xxx = []
# yyy = []
# zzz = []
# for grad in ekstremumai:
#     xxx.append(grad[0])
#     yyy.append(grad[1])
#     zzz.append(f(grad))
