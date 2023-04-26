import numpy
import random
from pprint import pprint

W1=random.uniform(0, 1)
W2=random.uniform(0, 1)

X1=[0,0,1,1]
X2=[0,1,0,1]
Y_and=[a*b for a,b in zip(X1,X2)]

SMQ_ALL=[[0 for i in range(21)] for j in range(21)]

for W1 in range(-10,11,1):
    for W2 in range(-10,11,1):
        
        Y=[a*W1/10+b*W2/10 for a,b in zip(X1,X2)]
        F=[0 if a<1 else 1 for a in Y]

        E2=[(a-b)**2 for a,b in zip(Y_and,F)]
        SMQ_ALL[W1+10][W2+10]=sum(E2)

pprint(SMQ_ALL)

