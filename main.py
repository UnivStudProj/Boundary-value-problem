import matplotlib.pyplot as plt
import numpy as np
import logging
import pandas as pd
from tkinter import *
from os import _exit

logging.basicConfig(filename="ignore/sample.log", format='%(message)s', level=logging.INFO)

# Input values
D = 0.06
c = 0.5
H = 0.007
u_c = 0
l = 10
T = 40
x_amount = 50
t_amount = 50


# Logging thing
def log(smth):
    logging.info('=' * 50)
    logging.info(smth)
    logging.info('=' * 50)

def prepVars():
    global I, K, h_x, h_t, sq_a, sigma, x_dots, t_dots, psi_of_x, u_prev, u_new, p, q
    
    # Varibles for Crank-Nicolson scheme
    I = x_amount - 1
    K = t_amount - 1
    h_x = l / x_amount 
    h_t = T / t_amount
    sq_a = D / c
    sigma = (h_t * sq_a) / (2 * h_x ** 2)
    
    # Arguments for the Crank-Nicolson scheme
    x_dots = np.linspace(0, l, num=x_amount)
    t_dots = np.linspace(0, T, num=t_amount)
    psi_of_x = [u_c + (1 + np.cos(np.pi * x / l)) for x in x_dots]

    # Arrays with the dots values
    u_prev = np.array([*psi_of_x], dtype=float)
    u_new = np.empty_like(u_prev) 

    # Sweep coeffs arrays
    p = np.empty(len(x_dots), dtype=float)
    q = np.empty_like(p)

# Preparing new dots for the Tomas algorithm
def setNewDots():
    u_new[0] = u_prev[0] + 2 * sigma * (u_prev[1] - u_prev[0])
    for i in range(1, I):
        u_new[i] = u_prev[i] + sigma * (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1])
    u_new[I] = u_prev[I] + 2 * sigma * ((-1 - H * h_x) * u_prev[I] + u_prev[I])


# Sweep method for the Crank-Nicolson scheme
def TomasAlgorithm(mat_A):
    setNewDots()
    # The forward sweep consists of the computation of new coefficients
    p[0] = -mat_A[0, 1] / mat_A[0, 0]
    q[0] = u_new[0] / mat_A[0, 0]
    for i in range(1, I):
        p[i] = -mat_A[i, i + 1] / (mat_A[i, i] + p[i - 1] * mat_A[i, i - 1])
        q[i] = (u_new[i] - mat_A[i, i - 1] * q[i - 1]) / (mat_A[i, i] + p[i - 1] * mat_A[i, i - 1])
    p[I] = 0
    q[I] = (u_new[I] - mat_A[I, I - 1] * q[I - 1]) / (mat_A[I, I] + mat_A[I, I - 1] * p[I - 1])
    # The solution is then obtained by back substitution
    u_new[I] = q[I]
    for i in range(I, 0, -1):
        u_new[i - 1] = u_new[i] * p[i - 1] + q[i - 1]
    return u_new


# Setting matrix "A" values
def setMatrix_A():
    prepVars()
    # filling matrix
    mat_A = np.mat(np.empty((len(x_dots), len(x_dots)), dtype=float))
    mat_A[0, 0] = 1 + 2 * sigma
    mat_A[0, 1] = -2 * sigma
    for i in range(1, I):
        mat_A[i, i - 1] = -sigma
        mat_A[i, i] = 1 + 2 * sigma
        mat_A[i, i + 1] = - sigma
    mat_A[I, I - 1] = -2 * sigma
    mat_A[I, I] = 1 + 2 * sigma + 2 * sigma * H * h_x
    return mat_A
  

# Adding rows with substance concentration values
def Solution(mat_A):
    global u_prev
    mat_U = np.mat(np.empty((len(t_dots), mat_A.shape[1]), dtype=float))
    mat_U[K] = psi_of_x
    for k in range(K - 1, -1, -1):
        mat_U[k] = TomasAlgorithm(mat_A)
        u_prev = np.array([*u_new], dtype=float)
    u_prev = np.array([*psi_of_x], dtype=float)
    return mat_U


# Stability analysis
def isStable():
    return sigma <= 1 / (2 + 2 * h_x * H)
    
  
# Truncation error analysis 
def computeErrorAnalysis():
    """ Performing the error analysis

        - Node: u(0, K)
        - Steps of discretization: 4
        - Adding 2 extra elements to some arrays,
            since it need to write the next two values of the node
            
        Loop logic
        ----------
        1. Set current dots amount and compute the first vector (U1)
        2. Divide the step by 2 (h_x / 2, h_t / 2)
        3. Compute the second vector (U2)
        4. Calculate the difference between last two vectors (U1 - U2)
        5. Do the 2, 3 and 4 steps one more time
        6. Calculate the small delta by dividing each other values obtained in step 4 (first one / second one)
        7. Multiply the step by 2 to set a new amount of dots for the next iteration 
        
    """
    I_arr = np.full(4, 50, dtype=int) # "I" values (immutable)
    K_arr = np.empty_like(I_arr) # "K" values (each value multiplied by 2)
    DeltaDiv2 = np.empty(6, dtype=float) # Dicretizating step in space/time by 2
    DeltaDiv4 = np.empty_like(DeltaDiv2) # Dicretizating step in space/time by 4
    SmallDelta = np.empty_like(DeltaDiv2) # The error
    for i in range(4):
        U1 = Solution(setMatrix_A())[0, 0]
        K_arr[i] = t_amount
        changeTimeInterval(1)
        U2 = Solution(setMatrix_A())[0, 0]
        DeltaDiv2[i] = U1 - U2
        changeTimeInterval(1)
        U3 = Solution(setMatrix_A())[0, 0]
        DeltaDiv4[i] = U2 - U3
        SmallDelta[i] = DeltaDiv2[i] / DeltaDiv4[i]
        changeTimeInterval()
    err = np.array([I_arr, K_arr[:4], DeltaDiv2[:4], DeltaDiv4[:4], SmallDelta[:4]], dtype=np.float16)
    createErrorTable(err)


# Showing the error analysis table
def createErrorTable(err):
    fig, ax = plt.subplots(figsize=(7, 3), dpi=122, num="The error table")
    ax.axis("tight")
    ax.axis("off")
    df = pd.DataFrame(err.T, columns=["I", "K", "Δhₓ / 2, Δhₜ / 2", "Δhₓ / 4, Δhₜ / 4", "δ"])
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    fig.tight_layout()
    plt.show()
   
   
# Increasing/Decreasing time interval 
def changeTimeInterval(mode=0):
    global t_amount, h_t, sigma, t_dots
    t_amount = t_amount * 2 if mode == 1 else t_amount // 2
    h_t = T / t_amount
    sigma = (h_t * sq_a) / (2 * h_x ** 2)
    t_dots = np.linspace(0, T, num=t_amount)
    
    
# Creating two plots 
def createPlots():
    U = Solution(setMatrix_A())
    # Creating one figure (window) which contains two axes (plots)
    fig, (ax1, ax2) = plt.subplots(
        nrows=1, # Number of rows of the subplot grid. 
        ncols=2, # Number of columns of the subplot grid. 
        figsize=(12, 5), # Figure size in inches (size also affected by dpi)
        num='Dynamic of substance concentretion change within the cylinder' # Window title
    )
    # Substance concentration by time plot
    for i in range(U.shape[0] - 1, -1, int(-U.shape[0] / 10)):
        ax1.plot(x_dots, np.ravel(U[i]), label=f'u(x, {T - i * h_t:.2f})')
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax1.set_title('In time')
    ax1.set_xlabel('Coords')
    ax1.set_ylabel('Substance concentration')
    ax1.grid()
   # Substance concentration by space plot
    for k in range(0, U.shape[1], int(U.shape[1] / 10)):
        ax2.plot(t_dots, np.flip(np.ravel(U[:, k])), label=f'u({k * h_x:.2f}, t)')
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax2.set_title('In space')
    ax2.set_xlabel('Time')
    ax2.grid()
    fig.tight_layout(w_pad=2) # Plots padding (width)
    plt.show()
    

# Shows lines with a different amount of "t" dots 
def createConvergencePlot():
    U = Solution(setMatrix_A())
    fig, ax = plt.subplots(figsize=(8, 5), num='The convergence plot')
    for _ in range(4):
        ax.plot(x_dots, np.ravel(U[0]), label=f'u(x, {T}), x = {x_amount}, t = {t_amount}')
        changeTimeInterval(1)
        U = Solution(setMatrix_A())
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_title('Dynamic of substance concentretion change\n when discretizating the grid')
    ax.set_xlabel('Coords')
    ax.set_ylabel('Substance concentration')
    ax.grid()
    fig.tight_layout()
    plt.show()
 

class App(Frame):
    
    INN_COLOR = '#2e2e2e' 
    BAR_COlOR = '#424242'
    TEXT_COLOR = '#e5e5e5'
    VAR_COLOR = '#f72585'
    TITLES = (('Length of a rod', 'l'), ('Time interval', 'T'), ('Dots in space', 'x'), 
              ('Dots in time', 't'), ('Diffusion factor', 'D'), ('Porosity factor', 'c'), ('Membrane diffusion factor', 'H'))
    innFrames_list = []
    widgetEntries = []
    varEntries = []
    BUTTONS = ('Solution plots', 'The error table', 'The convergence plot')
    currentValues = [l, T, x_amount, t_amount, D, c, H] 
    x = None
    y = None
    
    # Initializing the main frame
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.pack(fill=BOTH)

        self.create_widgets()

    # Creating widgets
    def create_widgets(self):
        self.borderFrame = Frame(self, width=500, height=600, bg=self.BAR_COlOR, highlightthickness=2, highlightbackground=self.VAR_COLOR, highlightcolor=self.VAR_COLOR)
        self.borderFrame.pack_propagate(False)
        self.borderFrame.pack(side=TOP)

        holderFrame = Frame(self.borderFrame, width=500, height=560, bg=self.INN_COLOR, relief='raised')
        holderFrame.pack_propagate(False)
        holderFrame.pack(side=BOTTOM)

        # Title
        self.title_name = Label(self.borderFrame, text='Boundary value problem', bg=self.BAR_COlOR, fg=self.TEXT_COLOR, font='Helvetica 15', anchor='w')
        self.title_name.pack(side=LEFT, expand=1, fill=BOTH, padx=5)

        # Close button
        self.close_btn = Label(self.borderFrame, width=5, text='x', bg=self.BAR_COlOR, fg=self.TEXT_COLOR, font='Aria 18')
        self.close_btn.pack(side=RIGHT)
        
        # Minimize button
        self.minimize_btn = Label(self.borderFrame, width=5, text='—', bg=self.BAR_COlOR, fg=self.TEXT_COLOR, font='Arial 18')
        self.minimize_btn.pack(side=RIGHT)
        
        # Validating entered symbols 
        def validate(*args):
            """Input validate function
            
                args:
                    action, index, value_if_allowed,
                    prior_value, text, validation_type, 
                    trigger_type, widget_name
                 
            """
            if args[2]:
                try:
                    float(args[2])
                    return True
                except ValueError:
                    return False
            else:
                return False
        
        validate_config = (holderFrame.register(validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        
        def setVars(cmd):
            if cmd == 0:
               createPlots()
            elif cmd == 1:
                computeErrorAnalysis()
            elif cmd == 2:
                createConvergencePlot()
            setFromInputs()
         
        # Building widget classes 
        def widgetProperties(wType, *TextProps, valType=None, wFrame=None, hl_th=1, hl_bg='#4d4f55', cmd=None):
            if wType == 'Frame':
                return Frame(holderFrame, width=250, height=50, bg=self.INN_COLOR, highlightthickness=hl_th, highlightbackground=hl_bg)
            elif wType == 'Label':
                return Label(innFrame, text=TextProps[0], bg=self.INN_COLOR, fg=TextProps[1], font=TextProps[2])
            elif wType == 'Entry':
                return Entry(self.innFrames_list[wFrame], width=7, textvariable=valType, bg=self.BAR_COlOR, fg='cyan', font=('Quant Antiqua', 15), justify=CENTER, validate='key', validatecommand=validate_config)
            elif wType == 'Button':
                return Button(wFrame, text=TextProps[0], command=lambda : setVars(cmd), bg='#161616', fg=self.TEXT_COLOR, font=('Comic Sans MS', 17), relief=GROOVE, bd=False)
        
        # Tracing input size 
        def traceLen(event):
            txtVar = self.varEntries[self.widgetEntries.index(self.focus_get())]
            txtVar.trace('w', lambda *args: characters_limit(txtVar))
           
        def setFromInputs(event=None):
            global l, T, x_amount, t_amount, D, c, H
            l = getint(self.varEntries[0].get())
            T = getint(self.varEntries[1].get())
            x_amount = getint(self.varEntries[2].get())
            t_amount = getint(self.varEntries[3].get())
            D = getdouble(self.varEntries[4].get())
            c = getdouble(self.varEntries[5].get())
            H = getdouble(self.varEntries[6].get())
        
        # Creating inscriptions and inputs 
        for l in self.TITLES: 
            innFrame = widgetProperties('Frame') 
            innFrame.pack(side=TOP, padx=10, pady=5, fill='x')
            self.innFrames_list.append(innFrame)
            
            name = widgetProperties('Label', l[0], self.TEXT_COLOR, ('Arial', 18))
            name.pack(side=LEFT)
        
            symbol = widgetProperties('Label', l[1], self.VAR_COLOR, ('Quant Antiqua', 17))
            symbol.pack(side=LEFT)
            
            if self.TITLES.index(l) >= 4:
                varTxt = DoubleVar(value=self.currentValues[self.TITLES.index(l)])
            else:
                varTxt = IntVar(value=self.currentValues[self.TITLES.index(l)])
            
            varWidget = widgetProperties('Entry', wFrame=self.TITLES.index(l), valType=varTxt) 
            varWidget.pack(side=RIGHT)
            varWidget.bind('<FocusIn>', traceLen)
            varWidget.bind('<KeyRelease>', setFromInputs)

            self.widgetEntries.append(varWidget)
            self.varEntries.append(varTxt)
        
        # Creating buttons 
        for b in self.BUTTONS:
            btnFrame = widgetProperties('Frame', hl_th=2, hl_bg=self.VAR_COLOR)
            btnFrame.pack(side=TOP, expand=True)
            
            btn = widgetProperties('Button', b, wFrame=btnFrame, cmd=self.BUTTONS.index(b)) 
            btn.pack(side=TOP)
            
        
        # Setting character limit 
        def characters_limit(entry_text):
            if len(str(entry_text.get())) > 0:
                entry_text.set(str(entry_text.get())[:4])

        # Event trigger when mouse on the "minimize" button 
        def hoverMinBtn(event):
            event.widget.config(bg='#272727')
       
        # Event trigger when mouse off the "minimlize" button 
        def unhoverMinBtn(event):
            event.widget.config(bg=self.BAR_COlOR)
            
        # Binding events to the "minimmize" button  
        self.minimize_btn.bind('<Enter>', hoverMinBtn)
        self.minimize_btn.bind('<Leave>', unhoverMinBtn)
        self.minimize_btn.bind('<Button-1>', self.minimize)
        
        # Event trigger when mouse on the "close" button 
        def hoverCloseBtn(event):
            event.widget.config(bg='#d00000')
        
        # Event trigger when mouse off the "close" button 
        def unhoverCloseBtn(event):
            event.widget.config(bg=self.BAR_COlOR)
       
        # Binding events to the "close" button 
        self.close_btn.bind('<Enter>', hoverCloseBtn)
        self.close_btn.bind('<Leave>', unhoverCloseBtn)
        self.close_btn.bind('<Button-1>', self.exitProgram)
        
        # Binding events to the title label 
        self.title_name.bind('<Button-1>', self.startMove)
        self.title_name.bind('<ButtonRelease-1>', self.stopMove)
        self.title_name.bind('<B1-Motion>', self.moving) 
        self.title_name.bind('<Map>', self.frame_mapped)
        
    def startMove(self, event):
        self.x = event.x
        self.y = event.y
        
    def stopMove(self, e):
        self.x = None
        self.y = None
        
    def moving(self, event):
        x = (event.x_root - self.x - self.borderFrame.winfo_rootx() + self.borderFrame.winfo_rootx())
        y = (event.y_root - self.y - self.borderFrame.winfo_rooty() + self.borderFrame.winfo_rooty())
        win.geometry(f'+{x}+{y}')
    
    def frame_mapped(self, e):
        win.update_idletasks()
        win.overrideredirect(True)
        win.state('normal')
        
    def minimize(self, e):
        win.update_idletasks()
        win.overrideredirect(False)
        win.state('iconic')
        
    def exitProgram(self, e):
        _exit(0)
        

# Program start
win = Tk()
win.geometry('500x600')
win.overrideredirect(True)

app = App(win)

# win.iconbitmap('ignore/vector.ico')

win.mainloop()
