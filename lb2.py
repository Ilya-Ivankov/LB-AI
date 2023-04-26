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
            #self.W+=(y-self.predict(x)).dot(x*abs(self.W.T)*self.speed_lern)
        self.SSE(Y,X)


def main():
    X = np.array([(1,1,0,0,0), (0,0,0,0,0),
                  (0,0,0,0,1), (0,0,1,1,1),
                  (1,1,0,1,0), (0,1,0,0,1),
                  (0,0,0,0,1), (0,1,1,0,0),
                  (1,1,0,0,1), (0,1,1,0,1),
                  (0,0,1,0,1), (1,1,1,1,1),
                  (1,1,1,0,1), (1,1,0,1,1),
                  (0,0,0,1,1), (1,0,0,1,1),
                  (0,0,1,0,0), (0,1,1,1,1),
                  (0,0,1,1,0), (1,0,0,0,0),
                  (1,0,0,0,1), (1,0,0,1,0),
                  (1,0,0,1,1), (1,0,1,0,0),
                  (1,1,1,1,0), (1,1,1,0,0)])
    X_test = np.array([(0,0,0,1,0), (0,1,1,1,0),
                       (1,1,0,1,0), (1,0,1,1,0)])
    
    Y=np.array([1 if sum(i)>2.5 else 0 for i in X])
    Y_test=np.array([1 if sum(i)>2.5 else 0 for i in X_test])

    ner=Neiro(len(X[0]),sigm)
    for i in range(100):
        ner.train_1(X,Y)
    print(ner.W)
    print(ner.predict(X))

    Y_=[1 if i>0.5 else 0 for i in ner.predict(X)]
    #print(Y_)
    #print(Y)
    print(Y-Y_)

    Y_=[1 if i>0.5 else 0 for i in ner.predict(X_test)]
    #print(Y_)
    #print(Y_test)
    print(Y_test-Y_)

    ner.visl_sse()

if __name__ == "__main__":
    main()
