import numpy as np
import random
import plotly.graph_objects as go

def sigm(x):
    return 1/(1+np.exp(-x))

class Neiro:
    
    def __init__(self, col_w, Ac, a=-1, b=1, sl=0.1):
        self.W=np.array([random.uniform(a,b) for i in range(col_w)])
        self.B=random.uniform(a,b)
        self.activ=Ac
        self.speed_lern=sl
        self.sse=[]

    def summ(self,X):
        return self.W.dot(X.T)+self.B

    def predict(self,X):
        return self.activ(self.summ(X))

    def SSE(self,Y,X):
        se=sum((Y-self.predict(X))**2)
        self.sse.append(se)

    def visl_sse(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(self.sse))), y=self.sse))
        fig.show()

    def train_1(self,X,Y):
        for x,y in zip(X,Y):
            self.W+=(y-self.predict(x))*(x*self.speed_lern)
            self.B+=(y-self.predict(x))*self.speed_lern
        self.SSE(Y,X)


def main():
    X = np.array([(0,0), (1,0),
                  (0,1), (1,1)])
    
    Y=np.array([X[i][0]^X[i][1] for i in range(len(X))])

    laer=[Neiro(len(X[0]),sigm),Neiro(len(X[0]),sigm)]
    out_laer=Neiro(len(laer),sigm)

    for i in range(100):
        
        Y_laer=[0,0]
        for j in range(len(laer)):
            Y_laer[j]=laer[j].predict(X)
            
        Y_laer=np.transpose(Y_laer)
        
        out_laer.train_1(Y_laer,Y)
        Y_laer_fen=[0,0]
        Y_laer=np.transpose(Y_laer)
        for i in range(len(Y_laer)):
            Y_laer_fen[i]=Y_laer[i]*out_laer.W[i]
        
        for j in range(len(laer)):
            laer[j].train_1(X,Y_laer_fen[j])

    Y_laer=[0,0]
    for j in range(len(laer)):
        Y_laer[j]=laer[j].predict(X)
        laer[j].visl_sse()
    Y_laer=np.transpose(Y_laer)
    Y_laer_fen=out_laer.predict(Y_laer)
    out_laer.visl_sse()

    print(X, ' ', Y_laer_fen)

if __name__ == "__main__":
    main()
