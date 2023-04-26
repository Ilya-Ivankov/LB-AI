import numpy as np
import random
import plotly.graph_objects as go

#метрики как считать растояние
def m_evcl(a, b):
    return sum((a-b)**2)**0.5

def m_mink(a, b, p=2):
    return sum((a-b)**p)**(1/p)

def m_menx(a, b):
    return sum(abs(a-b))

#класс конохина   
class grid_Kohon:
    def __init__(self, metr, sizes, gran):

        self.Metr=metr
        self.sizes=sizes
        self.W=[[random.uniform(gran[j][0], gran[j][1]) for j in range(self.sizes[0])] for i in range(self.sizes[1])]
        self.vis=[]

    #предскозание
    def predict(self, x):
        y=[]
        for w in self.W:
            y.append(self.Metr(x,w))
        return np.argmin(y)

    #обучение
    def fit(self, X, epoh, lr):

        for yer in range(epoh):

            for x in X:
                ind=self.predict(x)
                self.W[ind]+=lr*(x-self.W[ind])

            y=[[],[],[],[]]
            for p in X:
                y[g.predict(p)].append(p)
            for i in range(len(y)-1):
                y[i]=list(map(list, zip(*y[i])))
            y[len(y)-1]=list(map(list, zip(*self.W)))
            self.vis.append(y)
            
#визуализация
def visal(a,b):
    fig = go.Figure()
    col=['#0099ff', '#009900', '#0000ff', '#ff0000']

    fig.add_trace(go.Scatter(x=a[2][0], y=a[2][1], mode='markers', marker_color=col[0]))
    fig.add_trace(go.Scatter(x=a[1][0], y=a[1][1], mode='markers', marker_color=col[1]))
    fig.add_trace(go.Scatter(x=a[0][0], y=a[0][1], mode='markers', marker_color=col[2]))
    fig.add_trace(go.Scatter(x=b[0], y=b[1], mode='markers', marker_color=col[3]))

    fig.show()

def trec(gn,step_):
    frames=[]
    col=['#0099ff', '#009900', '#0000ff', '#ff0000']
    for j in range(gn):
        frames.append(go.Frame(name=str(j),data=[go.Scatter(visible=True, x=step_[j][0][0], y=step_[j][0][1],mode='markers', marker_color=col[0]),
                                     go.Scatter(visible=True, x=step_[j][1][0], y=step_[j][1][1],mode='markers', marker_color=col[1]),
                                     go.Scatter(visible=True, x=step_[j][2][0], y=step_[j][2][1],mode='markers', marker_color=col[2]),
                                     go.Scatter(visible=True, x=step_[j][3][0], y=step_[j][3][1],mode='markers', marker_color=col[3])]))
                    
    fig = go.Figure(data=[go.Scatter(visible=True, x=step_[0][0][0], y=step_[0][0][1],mode='markers', marker_color=col[0]),
                          go.Scatter(visible=True, x=step_[0][1][0], y=step_[0][1][1],mode='markers', marker_color=col[1]),
                          go.Scatter(visible=True, x=step_[0][2][0], y=step_[0][2][1],mode='markers', marker_color=col[2]),
                          go.Scatter(visible=True, x=step_[0][3][0], y=step_[0][3][1],mode='markers', marker_color=col[3])])


    steps=[]
    for i in range(gn):
        step = dict(
            label = str(i),
            method = 'animate',  
            args = [[str(i)]],
        )
                    
        steps.append(step)

    sliders = [dict(
    currentvalue = {"prefix": "Графиков отображается: ", "font": {"size": 20}},
    len = 0.9,
    x = 0.1,
    pad = {"b": 10, "t": 50},
    steps = steps,
    )]
    fig.update_layout(title="Выводим графики по очереди",
                      xaxis_title="Ось X",
                      yaxis_title="Ось Y",
                      updatemenus=[dict(direction="left",
                                        pad = {"r": 10, "t": 80},
                                        x = 0.1,
                                        xanchor = "right",
                                        y = 0,
                                        yanchor = "top",
                                        showactive=False,
                                        type="buttons",
                                        buttons=[dict(label="►", method="animate", args=[None, {"fromcurrent": True}]),
                                                 dict(label="❚❚", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                                                   "mode": "immediate",
                                                                                                   "transition": {"duration": 0}}])])],
                      )

    fig.layout.sliders = sliders

    fig.frames=frames
    fig.show()

#берем стартовый набор        
X_all=list(map(list, zip(*np.load('dat.npy'))))
Y=np.load('tar.npy')

X=[X_all[1],X_all[3]]#тут можно вместо 3 и 2 можно поставить любые числа [0-3] выбираем два пораметро за 4 доступных
gran=[[min(X[0]),max(X[0])],[min(X[1]),max(X[1])]]

#
X=np.array(list(map(list, zip(*X))))
g=grid_Kohon(m_evcl, [2, 3], gran)
g.fit(X,100,0.01)

#предсказываем результаты 
y=[[],[],[]]
for p in X:
    y[g.predict(p)].append(p)

for i in range(len(y)):
    y[i]=list(map(list, zip(*y[i])))

w=list(map(list, zip(*g.W)))

#рисуем результат
trec(100,g.vis)
#visal(y,w)

