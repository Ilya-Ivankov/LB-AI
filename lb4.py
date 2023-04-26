import numpy as np
import random
import plotly.graph_objects as go

def sigm(x):
    return 1.0/(1.0+np.exp(-x))

def sigm_pros(x):
    return sigm(x)*(1-sigm(x))

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

        delta=(As[-1]-y)*self.activ_pros(Zs[-1])#-1 это последний элемент !!!!!!!!!
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
        self.sse.append(s[0][0])

    def visl_sse(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(self.sse))), y=self.sse))
        fig.show()

def main():
    pass

if __name__ == "__main__":
    main()

    X = np.array([[[0],[0]], [[1],[0]],
                  [[0],[1]], [[1],[1]]])
    
    Y=np.array([ [X[i][0]^X[i][1]] for i in range(len(X))])

    net=Network([len(X[0]),2,2,1], sigm, sigm_pros)

    #print(net.predict(X[0]))# для одного будет 2 одинаковых ответа для много два ряда

    net.fit(X,Y,200,6)

    net.visl_sse()
    for i in X:
        print(net.predict(i))
