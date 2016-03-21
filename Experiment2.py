from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class experiment2:

    # Load the data
    iris = datasets.load_iris()

    # Adjust the data
    x = iris.data
    y = iris.target

    # Fix one param_grid for each algorithm (Er beslutsträd algoritm, er Random Forest-implementering
    # DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier
    
    k_range = list(range(1, 100))
    param_grid_knn = dict(n_neighbors=k_range)

    # Add all param_grids to a single list
    param_grids = [param_grid_knn]

    # Add all the algorithms into a list of models
    models = [KNeighborsClassifier()]

    scores = ['accuracy', 'precision_weighted', 'recall_weighted']

    for i in range(len(models)):
        for score in scores:
            # Set n_jobs = -1 maybe?
            grid = GridSearchCV(models[i], param_grids[i], cv=10, scoring=score)
            grid.fit(x, y)
            print(score + " " + str(grid.best_params_))




if __name__ == '__main__':
    e = experiment2()
