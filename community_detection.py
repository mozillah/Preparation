#!/usr/bin/env python
# coding: utf-8

# # Community detection using NETWORKX 
# 

# In[25]:


import networkx.algorithms.community as nxcom
from matplotlib.pyplot import figure, text
import itertools
import numpy as np
import re
import math
import os
import re
import sqlite3
import sys
import networkx as nx 
import nxviz as nv
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.kclique import k_clique_communities
import json
print(f"Python version {sys.version}")
print(f"networkx version: {nx.__version__}")


# In[26]:


dataFrames =[]
nxGraph = nx.Graph()
filesFrame ={}


# In[27]:


def getKeysByValue(dictOfElements, valueToFind):

    key = math.nan
    listOfItems = dictOfElements.items()

    for item  in listOfItems:
        if valueToFind in item[1] :
            key = item[0]
    return  key
def FilesDataExtract():
    global filesFrame
    newdata = []
    for fileItem in filesFrame.items():
        print(fileItem)
        filedata = {        }
        filedata["filename"] = fileItem[0]
        dataFrame = fileItem[1]
        countsData = dataFrame["region"].value_counts().to_dict()
        filedata ["counts"] = countsData
        filedata["total"]= int(dataFrame[["region"]].count(axis=0, level=None, numeric_only=False))
        filedata['genderCounts'] = pd.crosstab(dataFrame["region"], dataFrame["gender"]).to_dict(orient="index")
        # filedata["genderCounts"] = dataFrame["gender"].value_counts().to_dict()
        newdata.append(filedata)
    JsonData = json.dumps(newdata)
    return JsonData

def findRegion(countryname,num):
    westRegion = [22,23,5]
    northRegion = [24,25,27]
    centralRegion=[1,2,3,4,6,7,8,9,10,11,12,13,14,21,26]
    northEastRegion =[19,20,28]
    eastRegion =[15,16,17,18]

    # central, north, north-east, east, west
    singaporeDistrict={
        1:[1, 2, 3,4, 5, 6],
        2:[7,8],
        3:[14, 15, 16],
        4:[9,10],
        5:[11, 12, 13],
        6:[17],
        7:[18, 19],
        8:[20,21],
        9:[22,23],
        10:[24, 25, 26, 27],
        11:[28, 29, 30],
        12:[31, 32, 33],
        13:[34, 35, 36, 37],
        14:[38, 39, 40, 41],
        15:[42, 43, 44, 45],
        16:[46, 47, 48],
        17:[49, 50, 81],
        18:[51, 52],
        19:[53, 54, 55, 82],
        20:[56, 57],
        21:[58, 59],
        22:[60, 61, 62, 63, 64],
        23:[65, 66, 67, 68],
        24:[69, 70, 71],
        25:[72, 73,74],
        26:[77, 78],
        27:[75, 76],
        28:[79, 80]
    }


    if countryname != "Singapore":
        return "other"
        
    postalCodenum = int(str(num)[:2])

    if math.isnan(postalCodenum):
        print(f"key:{postalCodenum } other")
        return "other"

    # print(int(str(num)[:2]))
    key  = getKeysByValue(singaporeDistrict, postalCodenum)
    
    if math.isnan(key):
        return "other"
    else:
        if key in westRegion :
            return "west"
        elif key in northRegion :
            return "north"
        elif key in centralRegion:
            return "central"
        elif key in northEastRegion:
            return "northeast"
        elif key in eastRegion:
            return "east" 
   


# In[28]:


files =[
    r"C:\Users\telomere\Desktop\Projects\Chartjs\checkProject\project\static\excel\Event A.xlsx",
    r"C:\Users\telomere\Desktop\Projects\Chartjs\checkProject\project\static\excel\Event B.xlsx",
    r"C:\Users\telomere\Desktop\Projects\Chartjs\checkProject\project\static\excel\Event C.xlsx",
    r"C:\Users\telomere\Desktop\Projects\Chartjs\checkProject\project\static\excel\Event D.xlsx",
    r"C:\Users\telomere\Desktop\Projects\Chartjs\checkProject\project\static\excel\Event E.xlsx",
    r"C:\Users\telomere\Desktop\Projects\Chartjs\checkProject\project\static\excel\Event F.xlsx",
    r"C:\Users\telomere\Desktop\Projects\Chartjs\checkProject\project\static\excel\Event G.xlsx",
    r"C:\Users\telomere\Desktop\Projects\Chartjs\checkProject\project\static\excel\Event H.xlsx",
]


# In[41]:


files


# In[47]:


from community import community_louvain


# In[50]:


dataFrames =[]
nxGraph = nx.Graph()
filesFrame ={}
for file in files:     #image will be the key 
    # upload_file = uploaded_files[0]# request.files['upload']
        # upload_file = uploaded_files[fileKey]
    data1 = pd.concat(pd.read_excel(file, sheet_name=None), ignore_index=True)
    data1.dropna(how='any')
    data1.rename(columns={'event ': 'event'}, inplace=True)
    eventName = data1["event"][0]
#     files_name.append(upload_file.filename)
    for i in range(0, len(data1)):
        # name=data1["name"][i]
        node = data1["name"][i]
        email = data1["email"][i]
        node = re.sub("[\"\']", "", node)
        # node =re.sub("[\"\']", "", node)
        email = re.sub("[\"\']", "", email)
        data1["name"][i] = node
        data1["email"][i] = email
        gender = data1["gender"][i]
        nric = data1["nric"][i]
        # if node == "Rost'om":
        # print(node)
        pc = data1["postal code"][i]
        phone = data1["phone"][i]
        con = data1["country"][i]
        if (node not in nxGraph.nodes()):
            nxGraph.add_node(node,gender=gender,email=email,nric=nric, postal_code=pc, phone=phone, country=con)
            nxGraph.add_edge(node, eventName)
        else:
            nxGraph.add_edge(node, eventName)

    if 'postal code' in data1.columns:
        data1["region"] = data1[['country', 'postal code']].apply(lambda x: findRegion(*x), axis=1)
        filesFrame[file] = data1
    dataFrames.append(data1)

    dataFinal = pd.concat(dataFrames, axis=0, ignore_index=True)
    # venDiagram = dict(dataFinal["event"].value_counts())
    node_df = pd.DataFrame(columns=["id","gender","nric","email", "postal_code", "phone", "country","edges_len"])
    event_df = pd.DataFrame(columns=["id","edges_len"])
    for node in nxGraph.nodes(data=True):
        if (len(node[1]) == 0):
            print(node[0])
            node_df = node_df.append({"id": node[0]  ,"gender":"None","nric": "None","postal_code": "None","email":"None", "phone":"None","country": "None"
                                         , "edges_len": len(nxGraph.edges(node[0]))}, ignore_index=True)
            event_df = event_df.append(
                {"id": node[0], "edges_len": len(nxGraph.edges(node[0]))}, ignore_index=True)
            continue
        else:
            node_df = node_df.append(
                {"id": node[0],"gender":node[1]["gender"],"nric":node[1]["nric"],"email":node[1]["email"], "postal_code": node[1]["postal_code"], "phone": node[1]["phone"],
                 "country": node[1]["country"], "edges_len": len(nxGraph.edges(node[0]))}, ignore_index=True)


    communities_partition = community_louvain.best_partition(nxGraph)

    degree_centrality = nx.degree_centrality(nxGraph)
    close_centrality = nx.closeness_centrality(nxGraph)
    betweenness_centrality = nx.betweenness_centrality(nxGraph)
    page_rank  = nx.pagerank(nxGraph)


    node_df["degree_centrality"] = node_df.apply(lambda row: degree_centrality[row.id], axis=1)
    node_df["close_centrality"] = node_df.apply(lambda row: close_centrality[row.id], axis=1)
    node_df["betweenness_centrality"] = node_df.apply(lambda row: betweenness_centrality[row.id], axis=1)
    node_df["page_rank"] = node_df.apply(lambda row : page_rank[row.id],axis=1)

    event_df["degree_centrality"] = event_df.apply(lambda row: degree_centrality[row.id], axis=1)
    event_df["close_centrality"] = event_df.apply(lambda row: close_centrality[row.id], axis=1)
    event_df["betweenness_centrality"] = event_df.apply(lambda row: betweenness_centrality[row.id], axis=1)
    event_df["page_rank"] = event_df.apply(lambda row : page_rank[row.id],axis=1)

    #     cmap = {
    #     0: 'maroon',
    #     1: 'teal',
    #     2: 'black',
    #     3: 'orange',
    #     4: 'green',
    #     5: 'yellow',
    #     6: 'blue',
    # }
    #     unique_coms = np.unique(list(communities_partition.values()))
    #     color = {i : ("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])) for i in range(len(unique_coms))}

    def group(num):
        return communities_partition[num]

    import random
    unique_coms = np.unique(list(communities_partition.values()))
    color = {i: ("#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])) for i in
             range(len(unique_coms))}
    columns_community = ["name", "email", "gender", "nric", "phone", "address", "postal code", "country", "region",
                         "color", "community_name", "community"]
    communities_data_frame = pd.DataFrame.from_dict(communities_partition, orient='index', columns=["community"])
    communities_data_frame.reset_index(inplace=True)
    communities_data_frame['index'] = communities_data_frame['index'].apply(lambda x: re.sub("[\"\']", "", x))
    communities_data_frame['color'] = communities_data_frame['community'].apply(lambda x: color[x])
    communities_data_frame['community_name'] = communities_data_frame['community'].apply(
        lambda x: "community_" + str(x))
    new_data_final = pd.merge(communities_data_frame, dataFinal, right_on='name', left_on='index', how='left',
                              suffixes=('', '__2'))
    new_data_final["name"] = new_data_final["index"]
    new_data_final.drop_duplicates(subset="name", keep='first', inplace=True)
    final_comm_df = new_data_final[columns_community]

    node_df["group"] = node_df[['id']].apply(lambda x: group(*x), axis=1)

    edges_df = nx.to_pandas_edgelist(nxGraph)
    finalColumns = list(dataFinal.columns)
    columns_new = ["edges_len", "degree_centrality", "close_centrality","betweenness_centrality","page_rank","group"]
    finalColumns.extend(columns_new)
    new_data_final = pd.merge(dataFinal, node_df, left_on='name', right_on='id', suffixes=('', '__2'))[finalColumns]
    DatabaseColumns = dataFinal.columns
#     new_data_final.to_sql(name='participants', if_exists='replace', con=conn, index=False)
#     node_df.to_sql(name='nodes', if_exists='replace', con=conn, index=False)
#     edges_df.to_sql(name='edges', if_exists='replace', con=conn, index=False)
#     event_df.to_sql(name='events', if_exists='replace', con=conn, index=False)
#     final_comm_df.to_sql(name='communities', if_exists='replace', con=conn, index=False)

#     print(files_name)


# In[ ]:





# In[31]:


G = nxGraph


# In[17]:


# df = pd.read_csv("final_data.csv",sep=';',error_bad_lines=False,dtype = 'str')


# # Network preparation
# 

# In[20]:


# G = nx.Graph()

# # Tags Edges
# for i, row in df.iterrows():
#         if i<10000:
#                 tagArray = row['tags'].split(',')
#                 tagsCount = 0
#                 for tag in tagArray:
#                         tagsCount += 1
#                         userID = row['id'].strip()
#                         G.add_edge(userID, tag.strip(), color='r', weight=1, hastag=row['tags'])

# # Friends Edges
# for i, row in df.iterrows():
#      if i<0:
#         friendsArray = row['friends'].split(',')
#         friendsCount = 0
#         for friend in friendsArray:
#              friendsCount += 1
#              userID = row['id']
#              G.add_edge(userID, friend, color='r', weight=1, hastag=row['tags'])

# colors = nx.get_edge_attributes(G,'color').values()
# weights = nx.get_edge_attributes(G,'weight').values()
# hastags = nx.get_edge_attributes(G,'hastag').values()


# # Community detection

# In[32]:


def set_node_community(G, communities):
        #Add community attributes
        for c, v_c in enumerate(communities):
            #print(v_c)
            for v in v_c:
                # Set community for each node
                G.nodes[v]['community'] = c + 1

def get_color(i, r_off=1, g_off=1, b_off=1):
        #Assign a color to a vertex.
        r0, g0, b0 = 0, 0, 0
        n = 16
        low, high = 0.1, 0.9
        span = high - low
        r = low + span * (((i + r_off) * 3) % n) / (n - 1)
        g = low + span * (((i + g_off) * 5) % n) / (n - 1)
        b = low + span * (((i + b_off) * 7) % n) / (n - 1)
        return (r, g, b)  
    
def set_edge_community(G):
        # Set internal/external edges attributes
        # f - From / t - To Index
        for f, t, in G.edges:
            if G.nodes[f]['community'] == G.nodes[t]['community']:
                G.edges[f, t]['community'] = G.nodes[f]['community']
            else:
                G.edges[f, t]['community'] = 0


# In[33]:


G


# In[34]:


len(communities)


# In[35]:


communities = sorted(nxcom.greedy_modularity_communities(G), key=len)
#communities = next(nxcom.girvan_newman(G))
#communities = nxcom.label_propagation_communities(G)

set_node_community(G, communities)
set_edge_community(G)


# In[36]:


labels = {}
counts = {}
# Set center nodes attributes
for c, community in enumerate(communities):
    subGraph = G.subgraph(community)
    if len(subGraph.nodes) > 10:
        #print("{}: {}".format(nx.center(subGraph)[0], len(subGraph.nodes)))
        labels[nx.center(subGraph)[0]] = nx.center(subGraph)[0]
        
        if len(subGraph.nodes) < 100 & len(subGraph.nodes) >= 40:
            counts[nx.center(subGraph)[0]] = len(subGraph.nodes)
        elif len(subGraph.nodes) < 40:
            counts[nx.center(subGraph)[0]] = 40
        else:
            counts[nx.center(subGraph)[0]] = 100


# In[76]:


# for node in G.nodes():
#     print(G.node)


# In[79]:


# G.nodes.data()


# In[ ]:


# 


# # Nodes preparation

# In[37]:


node_color = [get_color(G.nodes[i]['community']) for i in G.nodes]
external = [(f, t) for f, t in G.edges if G.edges[f, t]['community'] == 0]
internal = [(f, t) for f, t in G.edges if G.edges[f, t]['community'] > 0]
internal_color = [get_color(G.edges[i]['community']) for i in internal]


# In[21]:


pos = nx.spring_layout(G,k=0.1, iterations=150)
# #pos = nx.kamada_kawai_layout(G)


# In[22]:


len(internal)


# In[23]:


len(external)


# # Plot the network

# In[38]:


plt.figure(figsize=(12,8), dpi= 150, facecolor='w', edgecolor='k')

# External egdes & nodes
nx.draw(
        G, pos=pos, node_size=0,
        edgelist=external, edge_color="black", with_labels=False)

# Internal egdes & nodes
nx.draw( G, pos=pos, node_color=node_color, node_size = 20,
        edgelist=internal, edge_color=internal_color, with_labels=False)

# Labels
labs = nx.draw_networkx_labels(G,pos,labels,font_size=0,font_color='r')

for node in labs:
    (x,y) = pos[node]
    text(x, y, node, fontsize=counts[node]/4, ha='center', va='center',bbox=dict(facecolor=get_color(G.nodes[node]['community']), alpha=0.8,edgecolor=get_color(G.nodes[node]['community']), linewidth=0.0),color="white")


# In[40]:


labels


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




