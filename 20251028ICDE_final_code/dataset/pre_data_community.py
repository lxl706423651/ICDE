from __future__ import division
import graph_tool as gt
import graph_tool.inference as gti
import time
import copy
import itertools
import random
import math
import networkx as nx
import numpy as np
import time
import gc
datasets=[]
#edge_dict={}
f=open("./data/twitter.txt")
data=f.read()
f.close()
rows=data.split('\n')
for row in rows:
    split_row=row.split(' ')
    name=(int(split_row[0]),int(split_row[1]),float(split_row[2]))
    edge=(int(split_row[0]),int(split_row[1]))
    # print(name)
    #edge_dict[edge]=0
    datasets.append(name)

G=nx.DiGraph()
G.add_weighted_edges_from(datasets)   #根据数据集创建有向图

del data,rows,datasets
gc.collect()

node_list=list(G.nodes)
nodenum=sorted(node_list,reverse=True)[0]+1

total_nodes=len(G.nodes)#总节点数
total_edges=len(G.edges)#总边数
print('total_nodes:',total_nodes, 'total_edges:',total_edges)

t=time.time()
G_gt = gt.Graph(directed=True)
G_gt.add_vertex(nodenum)
for u, v in G.edges():
    G_gt.add_edge(G_gt.vertex(u), G_gt.vertex(v))

state = gti.minimize_blockmodel_dl(
    G_gt,
    state=gti.OverlapBlockState # 图  
)

blocks = state.get_overlap_blocks()
bv = blocks[0]
Bl = dict()
with open("./dataset/twitter/twitter_mem.txt", 'w') as f:
    for u in G_gt.vertices():
        f.write("{} {}\n".format(u, " ".join(map(str, list(bv[u])))))
print('cost time:',time.time()-t)