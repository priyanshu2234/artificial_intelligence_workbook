# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
    ----------
    datafile_name: str
        path to data file

    K: int
        number of clusters to use

    feature_names: list
        list of feature_names

    Returns
    ---------
    fig: matplotlib.figure.Figure
        the figure object for the plot

    axs: matplotlib.axes.Axes
        the axes object for the plot
    """
    # ====> insert your code below here
    # Read data from file
    data = np.genfromtxt(datafile_name, delimiter=',')


    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = data[i, j]
    # Create K-Means cluster model with 20 initializations
    cluster_model = KMeans(n_clusters=K, n_init=20)
    cluster_model.fit(data)
    cluster_ids = cluster_model.predict(data)

    # Recompute cluster distances
    distances = np.zeros((data.shape[0], K))
    for i in range(data.shape[0]):
        for k in range(K):
            distances[i, k] = np.sum((data[i] - cluster_model.cluster_centers_[k]) ** 2)

    # Create canvas and axes
    num_feat = data.shape[1]
    fig, ax = plt.subplots(num_feat, num_feat, figsize=(12, 12))
    plt.set_cmap('viridis')

    # Get colors for histograms
    hist_col = plt.get_cmap('viridis', K).colors

    # Loop over each pair of features
    for feature1 in range(num_feat):
        # Set axis labels
        ax[feature1, 0].set_ylabel(feature_names[feature1])
        ax[0, feature1].set_xlabel(feature_names[feature1])
        ax[0, feature1].xaxis.set_label_position('top')

        for feature2 in range(num_feat):
            # Data copying
            x_data = data[:, feature1].copy()
            y_data = data[:, feature2].copy()

            # Sorting for scatter plots
            x_data = x_data[np.argsort(cluster_ids)]
            y_data = y_data[np.argsort(cluster_ids)]

            if feature1 != feature2:
                # Scatter plot for different features
                ax[feature1, feature2].scatter(x_data, y_data, c=cluster_ids)
            else:
                # Sorting and splitting per cluster
                for k in range(K):
                    cluster_data = []
                    for i in range(len(cluster_ids)):
                        if cluster_ids[i] == k:
                            cluster_data.append(x_data[i])
                    # Plot histogram for each cluster
                    ax[feature1, feature2].hist(cluster_data, bins=20, color=hist_col[k], edgecolor='black')

    # Set title with username
    username = "Priyanshu"  # Replace with your UWE username
    fig.suptitle(f'Visualisation of {K} clusters by {username}', fontsize=16, y=0.925)

    # Save visualization
    fig.savefig('myVisualisation.jpg')

    return fig, ax
    # <==== insert your code above here
