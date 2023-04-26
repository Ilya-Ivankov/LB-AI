import numpy as np
import random
import plotly.graph_objects as go
import pygame

def sigm(x):
    return 1.0/(1.0+np.exp(-x))

def sigm_pros(x):
    return sigm(x)*(1-sigm(x))

def logs(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def logs_pros(x):
    return 1-logs(x)**2

class Network:
    def __init__(self, sizes, A, AP):

        self.activ=A
        self.activ_pros=AP
        
        self.N_layers = len(sizes)
        
        self.sse=[]
        
        self.B=[np.random.randn(i,1) for i in sizes[1:]]
        self.W=[np.random.randn(j, i) for i,j in zip(sizes[:-1], sizes[1:])]
        
    def predict(self, a):
        for b, w in zip(self.B, self.W):
            a = self.activ(np.dot(w, a)+b)
        return a

    def fit(self, X, Y, epoh, sl):
        self.sse=[]
        for yer in range(epoh):

            nabl_b = [np.zeros(b.shape) for b in self.B]
            nabl_w = [np.zeros(w.shape) for w in self.W]

            for x,y in zip(X,Y):
                nb, nw = self.backprop(x,y)
                nabl_b = [nb_+nab for nb_, nab in zip(nabl_b, nb)]
                nabl_w = [nw_+naw for nw_, naw in zip(nabl_w, nw)]

            self.B=[b-n_b*sl for b,n_b in zip(self.B, nabl_b)]
            self.W=[w-n_w*sl for w,n_w in zip(self.W, nabl_w)]
            
            self.SSE(Y, X)
            print('epochs №', yer, ' err = ', self.sse[-1])

    def backprop(self, x, y):
        nabl_b=[np.zeros(b.shape) for b in self.B]
        nabl_w=[np.zeros(w.shape) for w in self.W]
        
        As=[x]
        Zs=[]
        a=x

        for b, w in zip(self.B, self.W):
            z=np.dot(w, a)+b
            Zs.append(z)
            a=self.activ(z)
            As.append(a)

        delta=(As[-1]-y)*self.activ_pros(Zs[-1])
        nabl_b[-1]=delta
        nabl_w[-1]=delta.dot(As[-2].T)

        for i in range(2, self.N_layers):
            delta = self.W[1-i].T.dot(delta)*self.activ_pros(Zs[-i])
            nabl_b[-i]=delta
            nabl_w[-i]=delta.dot(As[-1-i].T)

        return nabl_b, nabl_w

    def SSE(self,y,x):
        s=0
        for i,j in zip(y,x):
            s+=(i-self.predict(j))**2 
        self.sse.append(sum(s)[0])

    def visl_sse(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(self.sse))), y=self.sse))
        fig.show()

def drav(Ex,Ans):
    width = 400 # ширина игрового окна
    height = 600 # высота игрового окна
    fps = 10 # частота кадров в секунду

    len_arr=[[(150,300),(150,200)],
             [(150,200),(250,200)],
             [(250,200),(250,300)],
             [(250,300),(250,400)],
             [(250,400),(150,400)],
             [(150,400),(150,300)],
             [(150,300),(250,300)]]

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("цифра")
    clock = pygame.time.Clock()

    running = True
    while running:
        clock.tick(fps)# пауза для синхронизации кадров

        for event in pygame.event.get():
            #провераем закрытие окна
            if event.type == pygame.QUIT:
                running = False

        screen.fill( (255, 255, 255) )

        for i in range(len(Ex)):
            if Ex[i][0]==1:
                pygame.draw.line(screen, (0, 0, 0), len_arr[i][0], len_arr[i][1], 4)
                f = pygame.font.Font(None, 75)
                text = f.render(str(Ans), 1, (0, 0, 0))
                screen.blit(text, (300, 300))

        pygame.display.flip() #выводит графику

        
    pygame.quit()
        

def main():
    pass

if __name__ == "__main__":
    main()
    X = np.array([[ [1],[1],[1],[1],[1],[1],[0] ],#0
                  [ [0],[0],[1],[1],[0],[0],[0] ],#1
                  [ [0],[1],[1],[0],[1],[1],[1] ],#2
                  [ [0],[1],[1],[1],[1],[0],[1] ],#3
                  [ [1],[0],[1],[1],[0],[0],[1] ],#4
                  [ [1],[1],[0],[1],[1],[0],[1] ],#5
                  [ [1],[1],[0],[1],[1],[1],[1] ],#6
                  [ [0],[1],[1],[1],[0],[0],[0] ],#7
                  [ [1],[1],[1],[1],[1],[1],[1] ],#8
                  [ [1],[1],[1],[1],[1],[0],[1] ]])#9
    
    Y=np.array( [ [[1] if i==j else [0] for j in range(len(X))] for i in range(len(X)) ])

    net=Network([len(X[0]),11,12,len(Y[0])], logs, logs_pros)

    net.fit(X,Y,10000,0.004)

    net.visl_sse()
    for i,j in zip(X,Y):
        pred=net.predict(i)
        pred=np.argmax(pred)
        print(pred, ' = ',np.argmax(j))
        #drav(i,pred)

    pre=[ [1],[0],[1],[1],[0],[1],[1] ]
    drav(pre,np.argmax(net.predict(pre)))

    
