import numpy as np
from tkinter import *
import math
import matplotlib.pyplot as plt

D = 0.06
c = 0.5
H = 0.007
u_c = 0
l = 10
T = 40
dots_x = 50
dots_t = 50
h_x = l / dots_x
h_t = T / dots_t
x = np.linspace(0, l, num=dots_x)
t = np.linspace(0, T, num=dots_t)
sigma = (h_t * D / c) / (2 * h_x ** 2)
edgeVal = 1 / (2 + 2 * h_x * H)


def printMatrix(matrix, name):
    print(name, "=")
    print(matrix)


def psi(loc_x):
    return u_c + (1 + math.cos(math.pi * loc_x / l))


def computeFreeVector(prev_u):
    new_u = np.ndarray.copy(np.asarray(prev_u))
    new_u[0] = prev_u[0] + 2 * sigma * (prev_u[1] - prev_u[0])
    I = len(prev_u) - 1
    for i in range(1, I):
        new_u[i] = prev_u[i] + sigma * (prev_u[i + 1] - 2 * prev_u[i] + prev_u[i - 1])
    new_u[I] = (1 - 2 * sigma - 2 * sigma * h_x * H) * prev_u[I] + 2 * sigma * prev_u[I - 1]
    return np.mat(new_u)


def fillMatrixA(loc_A):
    I = loc_A.shape[1] - 1
    loc_A[0, 0] = 1 + 2 * sigma
    loc_A[0, 1] = -2 * sigma

    for i in range(1, I):
        loc_A[i, i - 1] = -sigma
        loc_A[i, i] = 1 + 2 * sigma
        loc_A[i, i + 1] = -sigma
    loc_A[I, I - 1] = -2 * sigma
    loc_A[I, I] = 1 + 2 * sigma + 2 * sigma * H * h_x
    return loc_A


# Проверка
def isCorrectArray(loc_a):
    n = len(loc_a)

    for row in range(0, n):
        if len(loc_a[row]) != n:
            print('Не соответствует размерность')
            return False

    for row in range(1, n - 1):
        if abs(loc_a[row][row]) < abs(loc_a[row][row - 1]) + abs(loc_a[row][row + 1]):
            print('Не выполнены условия достаточности')
            return False

    if (abs(loc_a[0][0]) < abs(loc_a[0][1])) or (abs(loc_a[n - 1][n - 1]) < abs(loc_a[n - 1][n - 2])):
        print('Не выполнены условия достаточности')
        return False

    for row in range(0, len(loc_a)):
        if loc_a[row][row] == 0:
            print('Нулевые элементы на главной диагонали')
            return False
    return True


# Прогонка
def TDMA(loc_A, loc_f):
    rowNum = loc_A.shape[0]
    colNum = loc_A.shape[1]
    I = colNum - 1
    u = np.zeros(rowNum, dtype=float)
    p = np.zeros(colNum, dtype=float)
    q = np.zeros(colNum, dtype=float)

    # Прямой ход
    p[0] = -loc_A[0, 1] / loc_A[0, 0]
    q[0] = loc_f[0, 0] / loc_A[0, 0]

    for i in range(1, I):
        denominator = loc_A[i, i] + p[i - 1] * loc_A[i, i - 1]
        p[i] = -loc_A[i, i + 1] / denominator
        q[i] = (loc_f[0, i] - loc_A[i, i - 1] * q[i - 1]) / denominator
    p[I] = 0
    q[I] = (loc_f[0, I] - loc_A[I, I - 1] * q[I - 1]) / (loc_A[I, I] + loc_A[I, I - 1] * p[I - 1])
    u[I] = q[I]

    # Обратный ход
    for i in range(I, 0, -1):
        u[i - 1] = u[i] * p[i - 1] + q[i - 1]
    return u


def doSolve(loc_x, loc_t):
    loc_f = list(map(psi, loc_x))
    loc_A = np.mat(np.zeros((len(loc_x), len(loc_x)), dtype=float))
    loc_A = fillMatrixA(loc_A)
    K = len(loc_t) - 1
    loc_U = np.mat(np.zeros((len(loc_t), loc_A.shape[1]), dtype=float))
    loc_U[[K]] = loc_f
    if isCorrectArray(np.asarray(loc_A)):
        for k in range(K - 1, -1, -1):
            loc_f = TDMA(loc_A, computeFreeVector(loc_f))
            loc_U[[k]] = loc_f
    return loc_U


def showPlots(loc_U):
    global x, t
    plt.figure(figsize=(6, 6))
    plt.title("Динамика изменения концентрации\nвещества внутри цилиндра\nво времени")
    plt.grid(True)
    plt.xlabel("Координата")
    plt.ylabel("Концентрация вещества")
    for i in range(loc_U.shape[0] - 1, 0, int(-loc_U.shape[0] / 10)):
        fig = plt.subplot()
        fig.plot(np.asarray(x), np.ravel(loc_U[[i]]), linewidth=1, label='u(x, ' + ("%.2f" % (T - i * h_t)) + ')')
        fig.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()
    plt.figure(figsize=(6, 6))
    plt.title("Динамика изменения концентрации\nвещества внутри цилиндра\nв пространстве")
    plt.grid(True)
    plt.xlabel("Время")
    plt.ylabel("Концентрация вещества")
    for i in range(0, loc_U.shape[1], int(loc_U.shape[1] / 10)):
        fig = plt.subplot()
        fig.plot(np.asarray(t), np.flip(np.ravel(loc_U[:, i])), linewidth=1, label='u(' + ("%.2f" % (i * h_x)) + ', t)')
        fig.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.show()


def showConvergencePlots(loc_U):
    global x, t, T
    plt.figure(figsize=(8, 8))
    plt.title("Динамика изменения фиксированного временного слоя\nпри измельчении сетки")
    plt.grid(True)
    plt.xlabel("Координата")
    plt.ylabel("Концентрация вещества")
    for i in range(5):
        fig = plt.subplot()
        fig.plot(np.asarray(x), np.ravel(loc_U[[0]]), linewidth=1,
                 label='u(x, ' + ('%.2f' % T) +
                       ') x = ' + ('%.2f' % dots_x) +
                       ' t = ' + ('%.2f' % dots_t))
        fig.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        increaseSizeTwoTimes()
        loc_U = doSolve(x, t)
    plt.show()


def computeErr():
    global x, t, h_x, h_t, sigma, dots_t, dots_x
    first_dots_t = dots_t
    first_dots_x = dots_x
    for i in range(4):
        print("dots_t = " + str(dots_t))
        print("dots_x = " + str(dots_x))
        establishStability()
        U1 = doSolve(x, t)
        # showPlots(U1)
        increaseSizeTwoTimes()
        establishStability()
        U2 = doSolve(x, t)
        # showPlots(U2)
        increaseSizeTwoTimes()
        establishStability()
        U3 = doSolve(x, t)
        # showPlots(U3)
        inc21 = U1[0, 0] - U2[0, 0]
        inc32 = U2[0, 0] - U3[0, 0]
        #inc21 = U1[first_dots_t * 2 ** i - 1 - 8, 0] - U2[first_dots_t * 2 ** (i + 1) - 1 - 16, 0]
        #inc32 = U2[first_dots_t * 2 ** (i + 1) - 1 - 16, 0] - U3[first_dots_t * 2 ** (i + 2) - 1 - 32, 0]
        #index1 = first_dots_x * 2 ** i - 1
        #index2 = first_dots_x * 2 ** (i + 1) - 1
        #index3 = first_dots_x * 2 ** (i + 2) - 1
        #inc21 = U1[0, first_dots_t - 1] - U2[0, first_dots_t - 1]
        #inc32 = U2[0, first_dots_t - 1] - U3[0, first_dots_t - 1]
        print(str(i) + ": inc21 = " + str(inc21))
        print(str(i) + ": inc32 = " + str(inc32))
        print(str(i) + ": delta = " + str(inc21 / inc32))
        decreaseSizeTwoTimes()


def increaseSizeTwoTimes():
    global dots_x, dots_t, x, t, h_x, h_t, sigma, edgeVal
    dots_t *= 2
    t = np.linspace(0, T, num=dots_t)
    h_t = T / dots_t
    # dots_x *= 2
    # x = np.linspace(0, l, num=dots_x)
    # h_x = l / dots_x
    sigma = (h_t * D / c) / (2 * h_x ** 2)
    edgeVal = 1 / (2 + 2 * h_x * H)


def decreaseSizeTwoTimes():
    global dots_x, dots_t, x, t, h_x, h_t, sigma, edgeVal
    dots_t = int(dots_t / 2)
    t = np.linspace(0, T, num=dots_t)
    h_t = T / dots_t
    # dots_x = int(dots_x / 2)
    # x = np.linspace(0, l, num=dots_x)
    # h_x = l / dots_x
    sigma = (h_t * D / c) / (2 * h_x ** 2)
    edgeVal = 1 / (2 + 2 * h_x * H)


def establishStability():
    global D, c, H, h_x, h_t, sigma, edgeVal
    print(str(sigma) + " " + ("<" if sigma < edgeVal else ">=") + " " + str(edgeVal))
    print(sigma < edgeVal)


class MyGui(object):
    def __init__(self):
        self.gui = None
        self.txt = None
        self.txt3 = None
        self.txt4 = None
        self.txt5 = None
        self.txt6 = None
        self.txt7 = None
        self.txt8 = None
        self.sp = None
        self.v1 = None
        self.v2 = None
        self.show()

    def show(self):
        self.gui = Tk()
        self.gui.title("Графический интерфейс программы")
        self.fill_frame()
        self.gui.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.loop()

    def loop(self):
        self.gui.mainloop()

    def clicked1(self):
        global l, T, dots_x, dots_t, x, t, h_x, h_t, sigma, edgeVal, D, c, H
        l = getint(self.txt.get())
        T = getint(self.txt3.get())
        D = getdouble(self.txt6.get())
        c = getdouble(self.txt7.get())
        H = getdouble(self.txt8.get())
        dots_x = getint(self.txt4.get())
        dots_t = getint(self.txt5.get())
        h_x = l / dots_x
        h_t = T / dots_t
        x = np.linspace(0, l, num=dots_x)
        t = np.linspace(0, T, num=dots_t)
        sigma = (h_t * D / c) / (2 * h_x ** 2)
        edgeVal = 1 / (2 + 2 * h_x * H)
        establishStability()
        U = doSolve(x, t)
        showPlots(U)
        print("Done")

    def clicked2(self):
        global l, T, dots_x, dots_t, x, t, h_x, h_t, sigma, edgeVal, D, c, H
        l = getint(self.txt.get())
        T = getint(self.txt3.get())
        D = getdouble(self.txt6.get())
        c = getdouble(self.txt7.get())
        H = getdouble(self.txt8.get())
        dots_x = getint(self.txt4.get())
        dots_t = getint(self.txt5.get())
        h_x = l / dots_x
        h_t = T / dots_t
        x = np.linspace(0, l, num=dots_x)
        t = np.linspace(0, T, num=dots_t)
        sigma = (h_t * D / c) / (2 * h_x ** 2)
        edgeVal = 1 / (2 + 2 * h_x * H)
        computeErr()
        print("Done")

    def clicked3(self):
        global l, T, dots_x, dots_t, x, t, h_x, h_t, sigma, edgeVal, D, c, H
        l = getint(self.txt.get())
        T = getint(self.txt3.get())
        D = getdouble(self.txt6.get())
        c = getdouble(self.txt7.get())
        H = getdouble(self.txt8.get())
        dots_x = getint(self.txt4.get())
        dots_t = getint(self.txt5.get())
        h_x = l / dots_x
        h_t = T / dots_t
        x = np.linspace(0, l, num=dots_x)
        t = np.linspace(0, T, num=dots_t)
        sigma = (h_t * D / c) / (2 * h_x ** 2)
        edgeVal = 1 / (2 + 2 * h_x * H)
        U = doSolve(x, t)
        showConvergencePlots(U)
        print("Done")

    def fill_frame(self):
        self.gui.resizable(True, True)
        self.gui.title("Схема Кранка - Николсон")
        self.gui.geometry('520x290')
        lbl = Label(self.gui, text="Длина стержня L:")
        lbl.grid(column=0, row=0, sticky="nsew")
        self.txt = Entry(self.gui, width=5)
        self.txt.insert(END, 10)
        self.txt.grid(column=1, row=0, sticky="nsew")
        lbl3 = Label(self.gui, text="Промежуток времени T:")
        lbl3.grid(column=0, row=1, sticky="nsew")
        self.txt3 = Entry(self.gui, width=5)
        self.txt3.insert(END, 40)
        self.txt3.grid(column=1, row=1, sticky="nsew")
        lbl4 = Label(self.gui, text="Мелкость дробления по пространству:")
        lbl4.grid(column=0, row=2, sticky="nsew")
        self.txt4 = Entry(self.gui, width=5)
        self.txt4.insert(END, 50)
        self.txt4.grid(column=1, row=2, sticky="nsew")
        lbl5 = Label(self.gui, text="Мелкость дробления по времени:")
        lbl5.grid(column=0, row=3, sticky="nsew")
        self.txt5 = Entry(self.gui, width=5)
        self.txt5.insert(END, 50)
        self.txt5.grid(column=1, row=3, sticky="nsew")
        lbl6 = Label(self.gui, text="Коэффициент диффузии D:")
        lbl6.grid(column=0, row=4, sticky="nsew")
        self.txt6 = Entry(self.gui, width=5)
        self.txt6.insert(END, 0.06)
        self.txt6.grid(column=1, row=4, sticky="nsew")
        lbl7 = Label(self.gui, text="Коэффициент пористости c:")
        lbl7.grid(column=0, row=5, sticky="nsew")
        self.txt7 = Entry(self.gui, width=5)
        self.txt7.insert(END, 0.5)
        self.txt7.grid(column=1, row=5, sticky="nsew")
        lbl8 = Label(self.gui, text="Мембранный коэффициент диффузии H:")
        lbl8.grid(column=0, row=6, sticky="nsew")
        self.txt8 = Entry(self.gui, width=5)
        self.txt8.insert(END, 0.007)
        self.txt8.grid(column=1, row=6, sticky="nsew")
        btn1 = Button(self.gui, text="Построить графики решения", command=self.clicked1)
        btn1.grid(column=0, row=7, sticky="nsew")
        btn2 = Button(self.gui, text="Рассчитать погрешность", command=self.clicked2)
        btn2.grid(column=0, row=8, sticky="nsew")
        btn3 = Button(self.gui, text="Построить график сходимости", command=self.clicked3)
        btn3.grid(column=0, row=9, sticky="nsew")

    def on_closing(self):
        self.gui.destroy()


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    MyGui()
