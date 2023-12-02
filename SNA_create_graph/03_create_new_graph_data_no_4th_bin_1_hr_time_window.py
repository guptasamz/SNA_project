# %%
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx
import pathlib

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

# %%
route_ids = [3,4,5,6,7,8,9,14,17,18,19,22,23,28,29,34,41,42,50,52,55,56]

# %%
all_df = []
for route_id in route_ids:
    inp_fp = f'/home/sgupta/WORK/DATASETS/WeGo_Bus_data/STOPLEVEL_PROCESSED/route_specific_w_census_dist_delay_weather_traffic/{route_id}'
    filename = f'stoplevel_route_{route_id}_w_census_dist_delay_weather_traffic.parquet'
    df = pd.read_parquet(f'{inp_fp}/{filename}')
    
    all_df.append(df)

all_df = pd.concat(all_df)

# %%
op_path = './data/graph_data/1_hr_time_window_dynamic_graph/'
pathlib.Path(op_path).mkdir(parents=True, exist_ok=True) 

# %%
# TODO: on all data. 
# Converting 'precipitation_intensity','temperature','humidity' scaling them between 0 and 1 for the model to understand these values 
num_columns = ['precipitation_intensity','temperature','humidity','actual_hdwy', 'delay','displacement','median_income_last12months','average_speed']
    

ss = MinMaxScaler()
ss.fit(all_df[num_columns])

import joblib
joblib.dump(ss,'./data/graph_data/1_hr_time_window_dynamic_graph/num_features.joblib')

# %%
def map_to_1_hour_window(window):
    window = (window // 2)

    if(window == 12):
        return 0
    else:
        return window 

# %%
del all_df

import gc
gc.collect()

# %%
apc_data = []
for route_id in route_ids:
    inp_fp = f'/home/sgupta/WORK/DATASETS/WeGo_Bus_data/STOPLEVEL_PROCESSED/graph_specific_data/{route_id}'
    filename = f'graph_ready_data_{route_id}_1_hr_time_window.parquet'

    df = pd.read_parquet(f'{inp_fp}/{filename}')

    apc_data.append(df)

apc_data = pd.concat(apc_data)

# %%
apc_data.sort_values(by=['transit_date','route_direction_name','departure_time'],inplace=True,ignore_index=True)


from sklearn.preprocessing import LabelEncoder
# ,'time_window_1_hr'
categorical_columns = ['dayofweek','month','year','time_window_1_hr']
for col in categorical_columns:
    le = LabelEncoder()
    le = le.fit(apc_data[col].unique())

    apc_data[f'{col}_cat'] = le.transform(apc_data[col])

# %% [markdown]
# ### Second Creating single graph for each time window - Dynamic Graph 
# 
# <!-- 2023-02-21	32.0	 -->

# %%
# Code plotting stuff for a single transit date - can be ignored for now 



import plotly.express as px
import pandas as pd

def plot_and_save_graph_with_mapbox(G,pos_df,transit_date,time_window):
    pos = pos_df.to_dict(orient='index')
    pos = {key: tuple(value.values()) for key, value in pos.items()}

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        # edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        # edge_y.append(None)

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    # Create a DataFrame or provide edge_x, edge_y, node_x, node_y
    # with the appropriate data for your plot.

    # Create the edge DataFrame (You can replace this with your actual data)
    edge_df = pd.DataFrame({'x': edge_x, 'y': edge_y})

    # Create the node DataFrame (You can replace this with your actual data)
    node_df = pd.DataFrame({'x': node_x, 'y': node_y, 'text': node_text})

    # Create the map
    fig = px.scatter_mapbox(node_df, lat="y", lon="x", hover_data="text",
                            color_discrete_sequence=["blue"], zoom=10,
                            center={"lat": 36.1627, "lon": -86.7816}, title="Network graph with Mapbox in Python")

    # Add the edges as lines (You can replace this with your actual data)
    for i in range(0, len(edge_df), 2):
        start_node = edge_df.iloc[i]
        end_node = edge_df.iloc[i + 1]
        line = pd.DataFrame({'x': [start_node['x'], end_node['x']], 'y': [start_node['y'], end_node['y']]})
        fig.add_trace(px.line_mapbox(line, lat="y", lon="x").data[0])

    # Customize the map layout
    fig.update_layout(mapbox_style="light",
                    mapbox_accesstoken="pk.eyJ1IjoiZ3VwdGFzYW16IiwiYSI6ImNsZ3d6Zzh0eTAwbjMzcW8wcnJybmp6cmcifQ.4ZGZIjNSFzk6aYjYUT3P1Q",  # Replace with your Mapbox access token
                    mapbox_center={"lat": 36.1627, "lon": -86.7816},
                    showlegend=False,
                    hovermode='closest',
                    margin={"b": 20, "l": 5, "r": 5, "t": 40},
                    annotations=[
                        dict(text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'>https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    op_fp = './plots/dynamic_graph_1_hr_time_window/graph_strucutre/with_mapbox'
    pathlib.Path(op_fp).mkdir(parents=True, exist_ok=True) 
    

    # Save or display the figure
    fig.write_html(f'{op_fp}/graph_with_mapbox_{transit_date}_{time_window}.html')

def plot_and_save_graph_without_mapbox(G,pos_df,transit_date,time_window):
    # pos = nx.spiral_layout(G)
    pos = pos_df.to_dict(orient='index')
    pos = {key: tuple(value.values()) for key, value in pos.items()}

    fig = plt.figure(1, figsize=(200, 80), dpi=60)
    nx.draw_networkx(G,font_size=100,node_size=30000, arrowsize=300, width=10,pos=pos)

    op_fp = './plots/dynamic_graph_1_hr_time_window/graph_strucutre/without_mapbox'
    pathlib.Path(op_fp).mkdir(parents=True, exist_ok=True) 


    plt.savefig(f'{op_fp}/graph_{transit_date}_{time_window}.pdf', dpi=300, bbox_inches='tight')

t = apc_data[(apc_data.transit_date == '2021-11-05')]

graphs = []
count = 0
time_window_count = 0
# Creating the road network graph for a single days data for now. Here each node represents the stop and edge represents the stop that the bus travels to.
# for (route_direction_name), tdf in tqdm(df.groupby(['route_direction_name' ])):   
for (transit_date, route_direction_name, time_window), tdf in tqdm(t.groupby(['transit_date', 'route_direction_name' , 'time_window_1_hr'])):   
    # display(tdf)
    if route_direction_name == 1:
        continue

    # print('RDN: ',route_direction_name)
    G=nx.from_pandas_edgelist(tdf, 'source', 'target', ['displacement','median_income_last12months',
                                                    'white_pct', 'black_pct', 'hispanic_pct', 'public_transit_pct',
                                                    'pct_public_transit_for_work'],create_using=nx.DiGraph())
    
    try:
        G.remove_node('DELETE_NODE')
    except:
        print('iteration:',count)

    graphs.append(G)
    
    pos_df = tdf.groupby(['stop_id']).agg({
                                            'map_longitude':'first','map_latitude':'first'
                                            })
    
    plot_and_save_graph_with_mapbox(G,pos_df,transit_date,time_window)

    plot_and_save_graph_without_mapbox(G,pos_df,transit_date,time_window)

# %%

graphs = []
count = 0
time_window_count = 0
# Creating the road network graph for a single days data for now. Here each node represents the stop and edge represents the stop that the bus travels to.
# for (route_direction_name), tdf in tqdm(df.groupby(['route_direction_name' ])):   
for (transit_date, route_direction_name, time_window), tdf in tqdm(apc_data.groupby(['transit_date', 'route_direction_name' , 'time_window_1_hr'])):   
    if route_direction_name == 1:
        continue

    # print('RDN: ',route_direction_name)
    G=nx.from_pandas_edgelist(tdf, 'source', 'target', ['displacement'],create_using=nx.DiGraph())
    
    try:
        G.remove_node('DELETE_NODE')
    except:
        print('iteration:',count)

    graphs.append(G)
    
    pos_df = tdf.groupby(['stop_id']).agg({
                                            'map_longitude':'first','map_latitude':'first'
                                            })

# %%
print(G.number_of_nodes())
print(G.number_of_edges())

# %% [markdown]
# ### Creating the feature node matrix for the first graph

# %%
def bin_load(load, capacity):
    percentages = [0., .33, .66, 1.0]
    # percentages = [0., .10, .25, 1.0]

    transit_cap = [round(p * capacity) for p in percentages]
    labels = [0, 1, 2]
    
    bin_label = pd.cut(x=[load], bins=transit_cap, labels=labels, include_lowest=True)[0]
    
    if pd.isnull(bin_label):
        return 3

    return bin_label

# %%
features = ['precipitation_intensity','temperature','humidity','delay','average_speed','extreme_congestion',
            'dayofweek','month','year','time_window_1_hr'] 
node_feature_matrix_graphs = []
y_class_graphs = []
# time_window_index = []

iteration = 0
for (transit_date, route_direction_name, time_window), tdf in tqdm(apc_data.groupby(['transit_date','route_direction_name' ,'time_window_1_hr'])):   
    # Create node feature matrix and the y_class matrix (Creating this as well as a dictionary for now)
    # Designed as a dictionary. Traverse each row sequentially and get the aggregate (mean) of all the values 
    node_feature_matrix = {}
    y_class = {}

    if route_direction_name == 1:
        continue

    # Setting the keys to maintain the sequential order with respect to nodes in graph
    for key in list(graphs[iteration].nodes):
        node_feature_matrix[key] = None
        y_class[key] = None

    for (stop_id), stop_df in tdf.groupby(['stop_id']):
        stop_id = stop_df['stop_id'].iloc[0]

        node_feature_matrix[stop_id] = {}
        # Getting the route_direction_name taking only the first value because this will be the same for all the stops in the graph (The graph have been grouped on this column)

        # Aggregate (mean) the Weather data - temp, precipitation, humidity. 
        node_feature_matrix[stop_id]['precipitation_intensity'] = stop_df.precipitation_intensity.mean()
        node_feature_matrix[stop_id]['temperature'] = stop_df.temperature.mean()
        node_feature_matrix[stop_id]['humidity'] = stop_df.humidity.mean()

        # node_feature_matrix[stop_id]['actual_hdwy'] = stop_df.actual_hdwy.mean()
        node_feature_matrix[stop_id]['delay'] = stop_df.delay.mean()

        # Aggregate Traffic data after merging.
        node_feature_matrix[stop_id]['average_speed'] = stop_df.average_speed.mean()
        node_feature_matrix[stop_id]['extreme_congestion'] = stop_df.extreme_congestion.mean()

        # Time features
        node_feature_matrix[stop_id]['dayofweek'] = stop_df.dayofweek_cat.iloc[0]
        node_feature_matrix[stop_id]['month'] = stop_df.month_cat.iloc[0]
        node_feature_matrix[stop_id]['year'] = stop_df.year_cat.iloc[0]
        node_feature_matrix[stop_id]['time_window_1_hr'] = stop_df.time_window_1_hr_cat.iloc[0] #Check whether this need to be one hot encoded 

        # Getting the binned load using mean load and mean vehicle capacity
        mean_load = stop_df.load.max()
        mean_vehicle_capacity = stop_df.vehicle_capacity.max()
        binned_load = bin_load(mean_load,mean_vehicle_capacity)

        # Getting the mean of the load - will bin it later using vehicle capacity
        y_class[stop_id] = binned_load

    node_feature_matrix_graphs.append(node_feature_matrix)
    y_class_graphs.append(y_class)
    iteration = iteration+1

    # time_window_index.append(time_window)


# %%

try:
    import pickle
    with open("./data/graph_data/1_hr_time_window_dynamic_graph/node_feature_matrix_13_features.pkl", 'wb') as fp:
        pickle.dump(node_feature_matrix_graphs, fp)

    with open("./data/graph_data/1_hr_time_window_dynamic_graph/y_class_13_features.pkl", 'wb') as fp:
        pickle.dump(y_class_graphs, fp)

    with open("./data/graph_data/1_hr_time_window_dynamic_graph/graph_13_features.pkl", 'wb') as fp:
        pickle.dump(graphs, fp)
except:
    print("Could not save!")

# %%
# import pickle 

# with open("./data/graph_data/1_hr_time_window_dynamic_graph/node_feature_matrix_10_features.pkl", 'rb') as fp:
#     node_feature_matrix_graphs = pickle.load(fp)


# with open("./data/graph_data/1_hr_time_window_dynamic_graph/y_class_10_features.pkl", 'rb') as fp:
#     y_class_graphs = pickle.load(fp)


# with open("./data/graph_data/1_hr_time_window_dynamic_graph/graph_10_features.pkl", 'rb') as fp:
#     graphs = pickle.load(fp)


# %%
# Converting y_class to list from dictionary
y_class = list(y_class.values())

# Converting node_feature_matrix to 2D matrix from dictionary of dictionaries
for key in node_feature_matrix:
    node_feature_matrix[key] = list(node_feature_matrix[key].values())
node_feature_matrix = list(node_feature_matrix.values())

# %%
# Below value should equal (number of nodes, number of features)
print("Feature Matrix shape:",np.array(node_feature_matrix).shape)
print("Number of nodes:",len(G.nodes))
print('Number of Features: ',len(features)) 

# %%
for iteration in tqdm(range(len(node_feature_matrix_graphs))):
    # Converting y_class to list from dictionary
    y_class_graphs[iteration] = list(y_class_graphs[iteration].values())

    # Converting node_feature_matrix to 2D matrix from dictionary of dictionaries
    for key in node_feature_matrix_graphs[iteration]:
        try:
            node_feature_matrix_graphs[iteration][key] = list(node_feature_matrix_graphs[iteration][key].values())
        except:
            print(key)
    node_feature_matrix_graphs[iteration] = list(node_feature_matrix_graphs[iteration].values())
        

# %%
# Print some stats
print("Number of graphs: ",len(graphs))
print("Number of node_feature_matrix_graphs: ",len(node_feature_matrix_graphs))
print("Number of y_class_graphs: ",len(y_class_graphs))


# %% [markdown]
# ### Converting the networkx graph to pygeometric

# %%
# Imputing values for None rows 

for i in range(len(node_feature_matrix_graphs)):
    for j in range(len(node_feature_matrix_graphs[i])):
        # print(node_feature_matrix_graphs[i][j])
        if(node_feature_matrix_graphs[i][j] is None):
            node_feature_matrix_graphs[i][j] = np.zeros(len(features))
        else:
            node_feature_matrix_graphs[i][j] = np.array(node_feature_matrix_graphs[i][j])


for i in range(len(y_class_graphs)):
    # print(y_class)
    for j in range(len(y_class_graphs[i])):
        if(y_class_graphs[i][j] is None):
            y_class_graphs[i][j] = 0

# %%
y_dict = {
    '0':0,
    '1':0,
    '2':0,
    '3':0,
    '4':0,
    'None':0
}

for i in range(len(y_class_graphs)):
    # print(y_class)
    for j in range(len(y_class_graphs[i])):
        y_dict[str(y_class_graphs[i][j])] += 1

y_dict

# {'0': 1014097, '1': 588765, '2': 83207, '3': 25561, '4': 0, 'None': 0}

# %%
# Creating the resulting dataframe with all the pygeometric graph data
dataset_all_routes = []

count = 0
for iteration in tqdm(range(len(graphs))):

    try:
        # Creating the py geometric graph from networkx graph
        pyg_graph = from_networkx(graphs[iteration],group_edge_attrs=['displacement'])
        # Setting the node feature matrix for the py geometric graph 
        pyg_graph.x = torch.tensor(node_feature_matrix_graphs[iteration]).float()
        # Setting the y_class for the py geometric graph 
        pyg_graph.y = torch.tensor(y_class_graphs[iteration]).long()

        dataset_all_routes.append(pyg_graph)
    except Exception as e:
        # print(e)
        count = count + 1
        continue
        # print("Some issue with graph: ",iteration)

print("These graphs were removed because there was only a single node in these graphs:",count)


# %%

data = dataset_all_routes[0]  # Get the first graph object.
# Hard coding for now the code after ends here can be used to derieve this
num_classes = 4
num_features = len(features)

print('====================')
print(f'Number of graphs: {len(dataset_all_routes)}')
print(f'Number of features: {num_features}')
print(f'Number of classes: {num_classes}')

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')



# %%
# Saving the final data
op_path = './data/graph_data/1_hr_time_window_dynamic_graph'
op_filename = 'dataset_all_route_dynamic_graph_10_features.pt'

torch.save(dataset_all_routes, f'{op_path}/{op_filename}')

print("DONE! Creating Data")


