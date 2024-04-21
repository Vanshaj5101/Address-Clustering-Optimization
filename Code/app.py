import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn import preprocessing, cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy
from scipy.spatial.distance import cdist
import math
import minisom
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from streamlit_folium import st_folium,folium_static

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from spectral_equal_size_clustering import SpectralEqualSizeClustering

location = [19.2294561, 72.8479905]

@st.cache_data
def load_data(pincode):
    df = pd.read_csv("../Data/mumbai_df_geocoded")
    df = df[df["formatted_addresss"].str.contains(str(pincode))]
    return df

@st.cache_data
def distance(origin, destination): 
    lat1, lon1 = origin[0],origin[1]
    lat2, lon2 = destination[0],destination[1]
    radius = 6371 # km
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c * 1000

    return d

def kmeans(data,k):
    model = cluster.KMeans(n_clusters=k, init='k-means++', 
                           max_iter=500, n_init=10, random_state=0)
    X = data[["lat","lng"]]
    km_X = X.copy()
    km_X["cluster"] = model.fit_predict(X)
    closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, 
                     km_X.drop("cluster", axis=1).values)
    km_X["centroids"] = 0
    for i in closest:
        km_X["centroids"].iloc[i] = 1
    km_df = data.copy()
    km_df[["cluster","centroids"]] = km_X[["cluster","centroids"]]

    #map
    x,y = "lat", "lng"
    color = "cluster"
    popup = "formatted_addresss"
    marker = "centroids"
    data = km_df.copy()

    lst_elements = sorted(list(km_df[color].unique()))
    lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in range(len(lst_elements))]
    data["color"] = data[color].apply(lambda x: lst_colors[lst_elements.index(x)])

    map_ = folium.Map(location=location, tiles="cartodbpositron",zoom_start=15)

    data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]], popup=row[popup],
           color=row["color"], fill=True,
           radius=8).add_to(map_),axis=1)
    
    legend_html = """<div style="position:fixed; bottom:10px; left:10px; 
    border:2px solid black; z-index:9999; 
    font-size:14px;">&nbsp;<b>"""+color+""":</b><br>"""

    for i in lst_elements:
        legend_html = legend_html+"""&nbsp;<i class="fa fa-circle 
        fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+"""">
        </i>&nbsp;"""+str(i)+"""<br>"""
    legend_html = legend_html+"""</div>"""
    map_.get_root().html.add_child(folium.Element(legend_html))

    lst_elements = sorted(list(km_df[marker].unique()))
    data[data[marker]==1].apply(lambda row: 
           folium.Marker(location=[row[x],row[y]], 
           popup=row[marker], draggable=False,          
           icon=folium.Icon(color="black")).add_to(map_), axis=1)

    st_map = st_folium(map_, width=1000,height=500,returned_objects=[])

def cluster_centroids(number_of_clusters,points):
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0, init='k-means++',max_iter=500,n_init=10).fit(points)
    
    id_label=kmeans.labels_    
    l_array = np.array([[label] for label in kmeans.labels_])
    clusters = np.append(points,l_array,axis=1)
    closest= scipy.cluster.vq.vq(kmeans.cluster_centers_, points)[0]
    centroids = np.array([])
    for i in closest:
        i = int(i)
        centroids = np.concatenate((centroids,points[i]),axis=0) # returns concatenation of size (number_of_clusters,0)
    centroids = centroids.reshape(number_of_clusters,2)
    return clusters,centroids

@st.cache_data
def create_data_model(matrix,depot):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = matrix
    data['num_vehicles'] = 1
    data['depot'] = depot
    return data

@st.cache_data
def optimal_k_using_tsp(points,threshold):
	value = 0
	for k in range(2,40):
		clusters, centroids = cluster_centroids(k,points)
		_, __, n_clust = clusters.max(axis=0)
		n_clust = int(n_clust)
		for cls_no in range(n_clust+1):
			cluster_i = clusters[clusters[:,2] == cls_no][:,np.array([True, True, False])] # 2d matrix for ith cluster
			distance_matrix_i = cdist(cluster_i,cluster_i, lambda ori,des: float((distance(ori,des))))
			for index in range(len(cluster_i)):
				if (cluster_i[index] == centroids[cls_no]).any():
					data = create_data_model(distance_matrix_i,index)
					manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                        data['num_vehicles'], data['depot'])
					routing = pywrapcp.RoutingModel(manager)
					def distance_callback(from_index,to_index):
						from_node = manager.IndexToNode(from_index)
						to_node = manager.IndexToNode(to_index)
						return data['distance_matrix'][from_node][to_node]
					transit_callback_index = routing.RegisterTransitCallback(distance_callback)
					routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
					search_parameters = pywrapcp.DefaultRoutingSearchParameters()
					search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
					solution = routing.SolveWithParameters(search_parameters)
					break
			if solution.ObjectiveValue()>threshold:
				value=0
				break
			else:
				value=1
				continue
		if value==0:
			continue
		if value==1:
			value = k
			break
	return value


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    st.write('Objective: {} metres'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for vehicle :\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    st.write(plan_output)
    plan_output += 'Route distance: {}meters\n'.format(route_distance)

@st.cache_data
def print_route(points,k):
    # for k=25, routing distance for each cluster, manual check for optimal k
    clusters, centroids = cluster_centroids(k,points)
    _, __, n_clust = clusters.max(axis=0)
    n_clust = int(n_clust)
    for cls_no in range(n_clust+1):
        cluster_i = clusters[clusters[:,2] == cls_no][:,np.array([True, True, False])] # 2d matrix for ith cluster
        distance_matrix_i = cdist(cluster_i,cluster_i, lambda ori,des: float((distance(ori,des))))
        for index in range(len(cluster_i)):
            if (cluster_i[index] == centroids[cls_no]).any():
                data = create_data_model(distance_matrix_i,index)


                # Create the routing index manager.
                manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                                    data['num_vehicles'], data['depot'])

                # Create Routing Model.
                routing = pywrapcp.RoutingModel(manager)


                def distance_callback(from_index, to_index):
                    """Returns the distance between the two nodes."""
                    # Convert from routing variable Index to distance matrix NodeIndex.
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    return data['distance_matrix'][from_node][to_node]

                transit_callback_index = routing.RegisterTransitCallback(distance_callback)

                # Define cost of each arc.
                routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

                # Setting first solution heuristic.
                search_parameters = pywrapcp.DefaultRoutingSearchParameters()
                search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

                # Solve the problem.
                solution = routing.SolveWithParameters(search_parameters)

                # Print solution on console.
                st.write("For "+str(cls_no)+" Cluster:")
                if solution:
                    print_solution(manager, routing, solution)
            
                #print("Centroid for "+str(cls_no)+" th cluster at "+str(index))
                break

def som(data,m,n):
    X = data[["lat","lng"]]
    map_shape = (m,n)
    ## scale data
    scaler = preprocessing.StandardScaler()
    X_preprocessed = scaler.fit_transform(X.values)
    ## clustering
    model = minisom.MiniSom(x=map_shape[0], y=map_shape[1], input_len=X.shape[1],random_seed = 0)
    model.train_batch(X_preprocessed, num_iteration=100, verbose=False)
    ## build output dataframe
    som_X = X.copy()
    som_X["cluster"] = np.ravel_multi_index(np.array([model.winner(x) for x in X_preprocessed]).T, dims=map_shape)
    ## find real centroids
    cluster_centers = np.array([vec for center in model.get_weights() for vec in center])
    closest, distances = scipy.cluster.vq.vq(cluster_centers, X_preprocessed)
    som_X["centroids"] = 0
    for i in closest:
        som_X["centroids"].iloc[i] = 1
    ## add clustering info to the som_df dataset
    som_df = data.copy()
    som_df[["cluster","centroids"]] = som_X[["cluster","centroids"]]


    #map
    x,y = "lat", "lng"
    color = "cluster"
    popup = "formatted_addresss"
    marker = "centroids"
    data = som_df.copy()

    lst_elements = sorted(list(som_df[color].unique()))
    lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in range(len(lst_elements))]
    data["color"] = data[color].apply(lambda x: lst_colors[lst_elements.index(x)])

    map_ = folium.Map(location=location, tiles="cartodbpositron",zoom_start=15)

    data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]], popup=row[popup],
           color=row["color"], fill=True,
           radius=8).add_to(map_),axis=1)
    
    legend_html = """<div style="position:fixed; bottom:10px; left:10px; 
    border:2px solid black; z-index:9999; 
    font-size:14px;">&nbsp;<b>"""+color+""":</b><br>"""

    for i in lst_elements:
        legend_html = legend_html+"""&nbsp;<i class="fa fa-circle 
        fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+"""">
        </i>&nbsp;"""+str(i)+"""<br>"""
    legend_html = legend_html+"""</div>"""
    map_.get_root().html.add_child(folium.Element(legend_html))

    lst_elements = sorted(list(som_df[marker].unique()))
    data[data[marker]==1].apply(lambda row: 
           folium.Marker(location=[row[x],row[y]], 
           popup=row[marker], draggable=False,          
           icon=folium.Icon(color="black")).add_to(map_), axis=1)

    st_map = st_folium(map_, width=1000,height=500,returned_objects=[])

def eqs_sc(data, k, eqf):
    X = data[["lat","lng"]]
    dist_tr = cdist(X,X, lambda ori,des: float((distance(ori,des))))
    eq_sclustering = SpectralEqualSizeClustering(nclusters=k,
                                         nneighbors=int(dist_tr.shape[0] * 0.1),
                                         equity_fraction=eqf,
                                         seed=0)

    model_eqs = eq_sclustering.fit(dist_tr)
    eqs_X = X.copy()
    eqs_X["cluster"] = model_eqs

    eqs_df = data.copy()
    eqs_df["cluster"] = eqs_X["cluster"]
    

    #map
    x,y = "lat", "lng"
    color = "cluster"
    popup = "formatted_addresss"
    data = eqs_df.copy()

    lst_elements = sorted(list(eqs_df[color].unique()))
    lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in range(len(lst_elements))]
    data["color"] = data[color].apply(lambda x: lst_colors[lst_elements.index(x)])

    map_ = folium.Map(location=location, tiles="cartodbpositron",zoom_start=15)

    data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]], popup=row[popup],
           color=row["color"], fill=True,
           radius=8).add_to(map_),axis=1)
    
    legend_html = """<div style="position:fixed; bottom:10px; left:10px; 
    border:2px solid black; z-index:9999; 
    font-size:14px;">&nbsp;<b>"""+color+""":</b><br>"""

    for i in lst_elements:
        legend_html = legend_html+"""&nbsp;<i class="fa fa-circle 
        fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+"""">
        </i>&nbsp;"""+str(i)+"""<br>"""
    legend_html = legend_html+"""</div>"""
    map_.get_root().html.add_child(folium.Element(legend_html))

    st_map = st_folium(map_, width=1000,height=500,returned_objects=[])
    
def main():
    st.title("Geo-Clustering of Mumbai Addresses")
    choose_pincode = st.sidebar.selectbox("Choose Pincode",
	['400008', '400026', '400010', '400027', '400005', '400001', '400021', '400020', '400080', '400082', '400067', '400101', 
	 '400064', '400024','400077', '400072', '400070', '400078', '400003', '400002', '400004', '400012', '400033', '400015', '400014', 
	 '400017', '400028', '400016', '400097', '400063', '400095', '400068', '400103', '400066', '400051', '400055', '400057', '400029', 
     '400047', '400089', '400050', '400098', '400025', '400018', '400013', '400052', '400054', '400037', '400022', '400031', '400062', 
     '400104', '400092', '400091', '400083', '400042', '400076', '400043', '400086', '400093', '400059', '400058', '400056', '400049', 
     '400071', '400102', '400053', '400019', '400036', '400006', '400034', '400069', '400030', '400061', '400007', '400075', '400065', 
     '400081', '400096', '400074', '400099', '400038', '400088', '400094', '400079', '400009', '400060', '400011', '400039', '400604', 
     '401107', '401202', '401201', '401105', '401106', '401101', '401209', '401208', '400084', '410201', '400087', '401303', '401104', 
     '401205', '360001', '401301', '401207', '400610', '982179', '401203', '400090', '401103', '986714', '421601', '401210', '421201', 
     '401304', '400601'])
    
    if choose_pincode:
        data = load_data(choose_pincode)
        st.write("Selected Pincode:",choose_pincode)
        st.write(data)

    points = np.array(data.loc[:,['lat','lng']])
    choose_model = st.sidebar.selectbox(
    "Select Model",
    [   "NONE",
        "Kmeans++ with TSP",
        "SOM",
        "Equal Size Spectral Clustering",
    
    ])

    if choose_model == "Kmeans++ with TSP":
        threshold = st.sidebar.number_input('Enter value for Threshold (in Meters): ',min_value=2000,value=3500)
        if threshold:
            threshold = int(threshold)
            k = optimal_k_using_tsp(points,threshold)
            st.write("Solution Validated using TSP Route Optimization for K: ",k)
        if k:
            k = int(k)
            kmeans(data,k)
            print_route(points,k)


    elif choose_model == "SOM":
        m = st.sidebar.number_input('Enter value for M',min_value=2,value=3)
        n = st.sidebar.number_input('Enter value for N',min_value=2,value=3)
        if m and n:
            st.write("Default Values Selected for M and N",m,"and",n)
            som(data,m,n)

    elif choose_model == "Equal Size Spectral Clustering":
        k = st.sidebar.number_input('Enter value for K(Number of equally distributed clusters expected): ',min_value=2,value=12)
        if k:
            eqf = st.sidebar.number_input('Enter value for Equity Fraction: ',min_value=0.5,max_value=1.0,value=0.8)
            if eqf:
                eqs_sc(data,k,eqf)
    
    # if choose_model == "Kmeans++ with TSP":
    #     k = st.sidebar.number_input('Enter value for K: ',min_value=2)
    #     if k:
    #         k = int(k)
    #         kmeans(data=data,k=k)
    #         pass


if __name__=="__main__":
    main()

