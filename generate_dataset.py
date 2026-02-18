import numpy as np
import networkx as nx
import itertools
import os

#generating graphs, if we choose only one then model will learn that generator’s parameterization, not curvature.

def generate_er(n):
    p = np.random.uniform(0.05, 0.3)
    return nx.erdos_renyi_graph(n, p)


def generate_ba(n):
    m = np.random.randint(1, 5)
    return nx.barabasi_albert_graph(n, m)


def generate_ws(n):
    k = np.random.randint(2, 6)
    beta = np.random.uniform(0.1, 0.5)
    return nx.watts_strogatz_graph(n, k, beta)


def generate_random_graph(n):
    graph_type = np.random.choice(["er", "ba", "ws"])

    if graph_type == "er":
        G = generate_er(n)
    elif graph_type == "ba":
        G = generate_ba(n)
    else:
        G = generate_ws(n)

    return G, graph_type

#computing features

def compute_features(G):
    n = G.number_of_nodes()

    degree_list = []
    for node, degree in G.degree():
        degree_list.append(degree)

    degree_array = np.array(degree_list)
    avg_degree = np.mean(degree_array)
    degree_variance = np.var(degree_array)

    clustering = nx.average_clustering(G)

    if nx.is_connected(G):
        graph_used = G
    else:
        largest_component = max(nx.connected_components(G), key=len)
        graph_used = G.subgraph(largest_component).copy()

    avg_path = nx.average_shortest_path_length(graph_used)
    diameter = nx.diameter(graph_used)

    adjacency_matrix = nx.to_numpy_array(G)
    adjacency_eigenvalues = np.linalg.eigvals(adjacency_matrix)
    spectral_radius = np.max(np.abs(adjacency_eigenvalues))

    laplacian_matrix = nx.laplacian_matrix(G).toarray()
    laplacian_eigenvalues = np.real(np.linalg.eigvals(laplacian_matrix))
    laplacian_eigenvalues = np.sort(laplacian_eigenvalues)

    if len(laplacian_eigenvalues) > 1:
        spectral_gap = laplacian_eigenvalues[1]
    else:
        spectral_gap = 0.0

    triangle_dict = nx.triangles(G)
    total_triangles = sum(triangle_dict.values()) / 3

    features = np.array([
        n,
        avg_degree,
        degree_variance,
        clustering,
        avg_path,
        diameter,
        spectral_radius,
        spectral_gap,
        total_triangles
    ])

    return features

#now calculating gromov delta value (trying to calculate exact value, if it crashes then will approx. it)

def compute_delta(G):
    length_dict = dict(nx.all_pairs_shortest_path_length(G))
    nodes = list(G.nodes())

    max_delta = 0

    for a, b, c, d in itertools.combinations(nodes, 4):
        Dab = length_dict[a][b]
        Dcd = length_dict[c][d]
        Dac = length_dict[a][c]
        Dbd = length_dict[b][d]
        Dad = length_dict[a][d]
        Dbc = length_dict[b][c]

        S1 = Dab + Dcd
        S2 = Dac + Dbd
        S3 = Dad + Dbc

        values = sorted([S1, S2, S3])
        delta = (values[2] - values[1]) / 2

        if delta > max_delta:
            max_delta = delta

    return max_delta

# generating dataset

def generate_dataset(num_graphs=300, n=40):

    X = []
    y = []
    graph_types = []

    for i in range(num_graphs):
        print(f"Generating graph {i+1}/{num_graphs}")

        G, graph_type = generate_random_graph(n)

        if not nx.is_connected(G):
            largest_component = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_component).copy()

        features = compute_features(G)
        delta = compute_delta(G)

        X.append(features)
        y.append(delta)
        graph_types.append(graph_type)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    graph_types = np.array(graph_types)

    return X, y, graph_types

if __name__ == "__main__":

    num_graphs = 300
    n = 40

    X, y, graph_types = generate_dataset(num_graphs=num_graphs, n=n)

    print("\nDataset statistics:")
    print("Min delta:", np.min(y))
    print("Max delta:", np.max(y))
    print("Mean delta:", np.mean(y))
    print("Std delta:", np.std(y))

    # Feature normalization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std

    # Train test split
    split_index = int(0.8 * len(X_normalized))

    X_train = X_normalized[:split_index]
    y_train = y[:split_index]

    X_test = X_normalized[split_index:]
    y_test = y[split_index:]

    # Save everything
    os.makedirs("data", exist_ok=True)

    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    np.save("data/X_mean.npy", X_mean)
    np.save("data/X_std.npy", X_std)
    np.save("data/graph_types.npy", graph_types)

    print("\nSaved dataset to 'data/' directory.")
