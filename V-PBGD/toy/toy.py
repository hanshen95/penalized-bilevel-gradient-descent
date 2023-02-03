import numpy as np
import yaml

def g(x,y):
    return (x+y)**2+x*(np.sin(x+y))**2

def dg(x,y):
    return 2*np.array([x+y+0.5*(np.sin(x+y))**2+x*np.sin(x+y)*np.cos(x+y),
                     x+y+x*np.sin(x+y)*np.cos(x+y)])

def f(x,y):
    return np.cos(4*y+2)/(1+np.exp(2-4*x))+0.5*np.log((4*x-2)**2+1)

def df(x,y):
    return np.array([4*np.exp(2-4*x)*np.cos(4*y+2)/(1+np.exp(2-4*x)**2)+(16*x-8)/((4*x-2)**2+1),
                     -4*np.sin(4*y+2)/(1+np.exp(2-4*x))])

def box(x,xlim):
    if x>max(xlim):
        return max(xlim)
    elif x<min(xlim):
        return min(xlim)
    else:
        return x

# v-pbgd algorithm
def solveF(x0,y0,alpha,gam,epsilon=0.01):
    ls=[float(np.abs(y0+x0))]  
    dF = df(x0,y0)+gam*dg(x0,y0)
    x_ = box(x0-alpha*dF[0],xlim)
    y_ = y0-alpha*dF[1]
    pg=1/alpha*np.array([x0-x_,y0-y_])
    pgs=[float(np.linalg.norm(pg))]
    k=0
    while np.linalg.norm(pg) > epsilon:
        x=x_
        y=y_
        dF = df(x,y)+gam*dg(x,y)
        x_ = box(x-alpha*dF[0],xlim)
        y_ = y-alpha*dF[1]
        pg=1/alpha*np.array([x-x_,y-y_])
        
        pgs.append(float(np.linalg.norm(pg)))
        ls.append(float(np.abs(y+x)))
        k+=1  
    return (x,y,k), (pgs, ls)


if __name__ == '__main__':
    
    xlim=[0,3.] # this is set C
    epsilon=1e-5 # stopping accuracy
    gams=[10] # penalty constant
    alpha0=0.1 # step size
    N=1000 # number of runs
    
    ps=np.zeros((N,2))
    for n in range(N):
        x=np.random.uniform(0,3.5)
        y=np.random.uniform(-5,8.5)
        # print('(x0,y0) is',(x,y))
        K=0
        for gam in gams:
            alpha=alpha0/gam
            (x,y,k), _ = solveF(x,y,alpha,gam,epsilon)
            K+=k  
        ps[n,0]=x
        ps[n,1]=y
        print('Run:',n, ', total steps K',K,', (x_K,y_K) is ({:.3f},{:.3f})'.format(x,y))
    log={"p": ps.tolist()}
    filename=f'./save/toy_gam{gams}_lr{alpha0}_eps{epsilon}_xlim{xlim}.yaml'
    file = open(filename, mode="w+")
    yaml.dump(log, file)
    file.close()
