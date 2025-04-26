from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object

    ax: matplotlib.axes.Axes
        axis
    """

    # ====> insert your code below here
    # Set up sizes to test
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Track results
    successes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    epochs = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(10)]

    # Test each size
    for i in range(10):
        size = sizes[i]

        # Run 10 trials
        for j in range(10):
            # Make model
            model = MLPClassifier(
                hidden_layer_sizes=(size,),
                max_iter=1000,
                alpha=0.0001,
                solver="sgd",
                learning_rate_init=0.1,
                random_state=j
            )

            # Train model
            model.fit(train_x, train_y)

            # Check accuracy
            acc = model.score(train_x, train_y)

            # Count success
            if acc == 1.0:
                successes[i] += 1
                epochs[i][j] = model.n_iter_

    # Calculate average epochs
    avg_epochs = []
    for i in range(10):
        sum_epochs = 0
        count = 0

        for j in range(10):
            if epochs[i][j] > 0:
                sum_epochs += epochs[i][j]
                count += 1

        if count > 0:
            avg_epochs.append(sum_epochs / count)
        else:
            avg_epochs.append(1000)

    # Make plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Success rate plot
    ax[0].plot(sizes, successes, marker='o')
    ax[0].set_title("Reliability")
    ax[0].set_xlabel("Hidden Layer Width")
    ax[0].set_ylabel("Success Rate")
    ax[0].set_xticks(sizes)

    # Epochs plot
    ax[1].plot(sizes, avg_epochs, marker='o')
    ax[1].set_title("Efficiency")
    ax[1].set_xlabel("Hidden Layer Width")
    ax[1].set_ylabel("Mean Epochs")
    ax[1].set_xticks(sizes)

    plt.tight_layout()
    # <==== insert your code above here

    return fig, ax
# make sure you have the packages needed
from approvedimports import *

#this is the class to complete where indicated
class MLComparisonWorkflow:
    """ class to implement a basic comparison of supervised learning algorithms on a dataset """

    def __init__(self, datafilename:str, labelfilename:str):
        """ Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.

        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        # Load the data files
        self.data_x = np.loadtxt(datafilename, delimiter=",")
        self.data_y = np.loadtxt(labelfilename, delimiter=",")
        # <==== insert your code above here

    def preprocess(self):
        """ Method to
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if there are more than 2 classes

           Remember to set random_state = 12345 if you use train_test_split()
        """
        # ====> insert your code below here
        # Split data into train and test sets
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, random_state=12345, stratify=self.data_y
        )

        # Normalize features
        min_vals = []
        max_vals = []

        # Find min and max for each feature
        for i in range(self.data_x.shape[1]):
            feature_values = self.data_x[:, i]
            min_vals.append(min(feature_values))
            max_vals.append(max(feature_values))

        # Normalize training data
        train_norm = []
        for row in self.train_x:
            norm_row = []
            for i in range(len(row)):
                if max_vals[i] == min_vals[i]:
                    norm_row.append(0)
                else:
                    norm_row.append((row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
            train_norm.append(norm_row)

        # Normalize test data
        test_norm = []
        for row in self.test_x:
            norm_row = []
            for i in range(len(row)):
                if max_vals[i] == min_vals[i]:
                    norm_row.append(0)
                else:
                    norm_row.append((row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
            test_norm.append(norm_row)

        # Convert to arrays
        self.train_x = np.array(train_norm)
        self.test_x = np.array(test_norm)

        # Create one-hot encoded labels if needed
        unique_classes = list(set(self.data_y))
        num_classes = len(unique_classes)

        # Check if we need one-hot encoding
        if num_classes > 2:
            # Prepare one-hot training labels
            train_onehot = []
            for label in self.train_y:
                onehot = [0] * num_classes
                class_index = unique_classes.index(label)
                onehot[class_index] = 1
                train_onehot.append(onehot)

            # Prepare one-hot test labels
            test_onehot = []
            for label in self.test_y:
                onehot = [0] * num_classes
                class_index = unique_classes.index(label)
                onehot[class_index] = 1
                test_onehot.append(onehot)

            # Store one-hot labels
            self.train_y_onehot = np.array(train_onehot)
            self.test_y_onehot = np.array(test_onehot)
        else:
            # Binary classification - no need for one-hot
            self.train_y_onehot = self.train_y
            self.test_y_onehot = self.test_y
        # <==== insert your code above here

    def run_comparison(self):
        """ Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.

        For each of the algorithms KNearest Neighbour, DecisionTreeClassifier and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination,
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.
        """
        # ====> insert your code below here
        # Import needed models
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        # KNN parameter tuning
        k_values = [1, 3, 5, 7, 9]
        for i, k in enumerate(k_values):
            # Create and train KNN model
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.train_x, self.train_y)
            self.stored_models["KNN"].append(knn)

            # Test accuracy
            predictions = knn.predict(self.test_x)
            correct = 0
            for j in range(len(self.test_y)):
                if predictions[j] == self.test_y[j]:
                    correct += 1
            accuracy = (correct / len(self.test_y)) * 100

            # Update best model if better
            if accuracy > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy
                self.best_model_index["KNN"] = i

        # Decision Tree parameter tuning
        depths = [1, 3, 5]
        splits = [2, 5, 10]
        leafs = [1, 5, 10]

        dt_index = 0
        for depth in depths:
            for split in splits:
                for leaf in leafs:
                    # Create and train Decision Tree
                    dt = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=split,
                        min_samples_leaf=leaf,
                        random_state=12345
                    )
                    dt.fit(self.train_x, self.train_y)
                    self.stored_models["DecisionTree"].append(dt)

                    # Test accuracy
                    predictions = dt.predict(self.test_x)
                    correct = 0
                    for j in range(len(self.test_y)):
                        if predictions[j] == self.test_y[j]:
                            correct += 1
                    accuracy = (correct / len(self.test_y)) * 100

                    # Update best model if better
                    if accuracy > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy
                        self.best_model_index["DecisionTree"] = dt_index

                    dt_index += 1

        # MLP parameter tuning
        first_layers = [2, 5, 10]
        second_layers = [0, 2, 5]
        activations = ["logistic", "relu"]

        mlp_index = 0
        for first in first_layers:
            for second in second_layers:
                for activation in activations:
                    # Set up layer sizes
                    if second == 0:
                        layers = (first,)
                    else:
                        layers = (first, second)

                    # Create and train MLP
                    mlp = MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation=activation,
                        max_iter=1000,
                        random_state=12345
                    )
                    mlp.fit(self.train_x, self.train_y_onehot)
                    self.stored_models["MLP"].append(mlp)

                    # Test accuracy
                    predictions = mlp.predict(self.test_x)
                    correct = 0

                    # Check if multiclass
                    if len(set(self.data_y)) > 2:
                        for j in range(len(self.test_y)):
                            # Find predicted class
                            pred_class = 0
                            max_val = predictions[j][0]
                            for k in range(1, len(predictions[j])):
                                if predictions[j][k] > max_val:
                                    max_val = predictions[j][k]
                                    pred_class = k

                            # Find true class
                            true_class = 0
                            max_val = self.test_y_onehot[j][0]
                            for k in range(1, len(self.test_y_onehot[j])):
                                if self.test_y_onehot[j][k] > max_val:
                                    max_val = self.test_y_onehot[j][k]
                                    true_class = k

                            # Check if correct
                            if pred_class == true_class:
                                correct += 1
                    else:
                        # Binary classification
                        for j in range(len(self.test_y)):
                            if predictions[j] == self.test_y[j]:
                                correct += 1

                    accuracy = (correct / len(self.test_y)) * 100

                    # Update best model if better
                    if accuracy > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy
                        self.best_model_index["MLP"] = mlp_index

                    mlp_index += 1
        # <==== insert your code above here

    def report_best(self):
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN","DecisionTree" or "MLP"

        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        # ====> insert your code below here
        # Find the best algorithm
        best_algo = ""
        best_acc = 0

        # Check each algorithm
        algos = ["KNN", "DecisionTree", "MLP"]
        for algo in algos:
            if self.best_accuracy[algo] > best_acc:
                best_acc = self.best_accuracy[algo]
                best_algo = algo

        # Get the best model
        best_model = self.stored_models[best_algo][self.best_model_index[best_algo]]

        # Return results
        return best_acc, best_algo, best_model
        # <==== insert your code above here
