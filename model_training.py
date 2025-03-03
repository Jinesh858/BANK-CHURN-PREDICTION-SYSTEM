import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_models(X, y):
    """Train multiple ML models and save the best one."""
    
    print("📊 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and their hyperparameters
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
    }

    # Hyperparameter tuning (Only for RandomForest and KNN for now)
    param_grid = {
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'KNN': {'n_neighbors': [3, 5, 7]}
    }

    best_models = {}
    accuracies = {}

    for name, model in models.items():
        print(f"🚀 Training {name} model...")
        if name in param_grid:
            grid = GridSearchCV(model, param_grid[name], cv=3, scoring='accuracy')
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        best_models[name] = best_model
        accuracies[name] = accuracy

        print(f"✅ {name} Accuracy: {accuracy:.4f}")

    # Select the best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = best_models[best_model_name]

    # Save the best model
    joblib.dump(best_model, "best_model.pkl")
    print(f"🏆 Best Model: {best_model_name} (Saved as 'best_model.pkl')")

    return best_model, accuracies
