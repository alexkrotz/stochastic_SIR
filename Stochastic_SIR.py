import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import sys
import os 


args = sys.argv[1:]
if not args:
	print('Usage: python3 Stochastic_SIR.py n m ntrials tmax')
	sys.exit(1)
n = int(args[0])
m = int(args[1])
ntrials = int(args[2])
tmax = int(args[3])
    
infec_rate = .05  ### probability of an infected person infecting their neighbor
infec_time = 14 ### average time (days) that an infected person will remain sick/infectious
def get_adj_nodes(n,adj_mat):
    return np.where(adj_mat[n] == 1)[1]
def infect(status, adj_mat):
    status_out = np.copy(status)
    for n in range(len(status)):
        if status[n] >= 1 and status[n] < infec_time: # infectious
            #print(n,status[n])
            neighbors = get_adj_nodes(n,adj_mat)
            #print(neighbors)
            for n in range(len(neighbors)):
                if status[neighbors[n]] == 0: # infectable
                    if np.random.rand() < infec_rate:
                        status_out[neighbors[n]] = 1
    return status_out
immun_loss_prob = .01 #probability of loosing immunity each day after immjun_loss_time
immun_loss_time = 60 #length of time a person has immunity (days)
def update_status(status):
    for n in range(len(status)):
       # if status[n] == infec_time:
       #     if np.random.rand() < reinfect:
       #         status[n] = 0
        if status[n] < infec_time and status[n] > 0:
            #status[n] += 1
            if status[n] * np.random.rand() > infec_time/2:
                status[n] = infec_time
            else:
                status[n] += 1
        if status[n] >= infec_time:
            status[n] += 1
        if status[n] >= immun_loss_time:
            if np.random.rand() < immun_loss_prob:
                status[n] = 0
    return status
    
def update_colors(status):
    colors = []
    for n in status:
        if n >= 1 and n < infec_time:
            colors.append('red')
        if n < 1 or n >= infec_time:
            colors.append('blue')
    return colors      
    
    
def get_i(status):
    i=0
    for n in status:
        if n != 0 and n < infec_time:
            i += 1
    return i

foldername = './n_'+str(n)+'_m_'+str(m)+'_'

def save_data(dataname,trial,data):
    out_name = foldername+dataname+'_'+str(trial)+'.dat'
    np.savetxt(out_name, data)
    return



for trial in tqdm(range(0,ntrials)):
    tdat = np.array([])
    sdat = np.array([])
    idat = np.array([])
    rdat = np.array([])
    status_array = np.zeros((tmax,n**2),dtype=int)
    g = nx.generators.random_graphs.barabasi_albert_graph(n**2,m)
    adj_mat = nx.linalg.graphmatrix.adjacency_matrix(g).todense()
    if trial%5 == 0:
        save_data('ADJ_MAT',str(trial),adj_mat)
    #adj_mat = adj_mat_nn
    #rows, cols = np.where(adj_mat== 1)
    #edges = zip(rows.tolist(), cols.tolist())
    #gr = nx.Graph()
    #gr.add_edges_from(edges)
    #nx.draw(gr, node_size=500, with_labels=True)
    #plt.show()
    #adj_mat = adj_mat*0 + 1
    #adj_mat = np.ones((n**2,n**2),dtype=int)
    #print(len(adj_mat))
    #print(g.nodes)
    #print(g.edges)
    
    status = np.zeros(n**2,dtype=int)
    #status[0] = 1
    status[np.random.randint(0,len(status))] = 1
    colors = update_colors(status)
    n2=0
    for t in np.arange(0,tmax,1):
        #print(t)
        #plt.scatter(coords_x,coords_y,c=colors)
        #plt.savefig('collision_pics/img_'+str(n2)+'.png')
        #plt.close()
        s = np.sum(status==0)
        i = get_i(status)
        r = np.sum(status >= infec_time)
        status_array[t] = status
        #print(s,i,r)
        #if i == 0:
        if t%100 == 0 and t > 0:
            for n3 in range(len(status)):
                if status[n3] ==0:
                    if np.random.rand() < .01:
                        status[n3] = 1
                
        tdat = np.append(tdat,np.array([t]))
        sdat = np.append(sdat,np.array([s]))
        idat = np.append(idat,np.array([i]))
        rdat = np.append(rdat,np.array([r]))
        #tdat.append(t)
        #sdat.append(s)
        #idat.append(i)
        #rdat.append(r)
        #print(status)
        status = infect(status,adj_mat)
        status = update_status(status)
        colors = update_colors(status)
        #if i == 0:
        #    break
    
    #print(status)
        
        n2+=1
    #plt.show()
    if trial%5 == 0:
        save_data('S',str(trial),sdat)
        save_data('I',str(trial),idat)
        save_data('R',str(trial),rdat)
        save_data('STATUS',str(trial),status_array)
    if trial == 0:
        save_data('T','',tdat)
        tdat_out = tdat
        sdat_out = sdat
        idat_out = idat
        rdat_out = rdat
    else:
        sdat_out += sdat
        idat_out += idat
        rdat_out += rdat
        
tot = sdat_out[0]
sdat_out = sdat_out/tot
idat_out = idat_out/tot
rdat_out = rdat_out/tot
save_data('S','tot',sdat_out)
save_data('I','tot',idat_out)
save_data('R','tot',rdat_out)

#plt.plot(tdat_out,sdat_out,label='Succeptible (%)')
#plt.plot(tdat_out,idat_out,label='Infected (%)')
#plt.plot(tdat_out,rdat_out,label='Recovered/Resistant (%)')
#plt.legend()
#plt.show()
