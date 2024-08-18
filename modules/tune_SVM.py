from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def tune_svm_model(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # Initialize the SVM model
    svm = SVC()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm, param_grid, refit=True, verbose=2, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_params, best_model
