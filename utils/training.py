from joblib import Parallel, delayed
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeavePOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import random
import xgboost
from sklearn.model_selection import RandomizedSearchCV
import sklearn
import time


def get_cv(X_train):
    # choose cross-validation strategy based on dataset size
    n_samples = X_train.shape[0]
    if n_samples <= 12:
        cv = LeavePOut(2)
    elif n_samples < 128:
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    else:
        # 20% validation or max 100 samples
        val_size = min(int(0.2 * n_samples), 100)
        train_size = n_samples - val_size
        cv = [(list(range(train_size)), list(range(train_size, n_samples)))]
    return cv


def get_splits(cv, X_train, y_train):
    # only keep splits where both classes appear in the validation set
    if hasattr(cv, 'split'):
        splits = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            if len(np.unique(y_train[val_idx])) == 2:
                splits.append((train_idx, val_idx))
    else:
        splits = cv
    return splits


def find_best_reg(X_train, y_train, X_test, y_test, plot=False, n_jobs=-1, parallel=False, penalty="l2", seed=1, return_classifier=False):
    # grid search over C values using cross-validation, then refit on full train set
    best_C = None
    if X_train.shape[0] > 3:
        cv = get_cv(X_train)
        Cs = np.logspace(5, -5, 10)
        avg_scores = []

        def evaluate_fold(C, train_index, val_index):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
            if penalty == "l1":
                model = LogisticRegression(C=C, penalty="l1", solver="saga", random_state=seed, max_iter=1000)
            else:
                model = LogisticRegression(C=C, random_state=seed, max_iter=1000)
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            return roc_auc_score(y_fold_val, y_pred_proba)

        for C in Cs:
            splits = get_splits(cv, X_train, y_train)
            if parallel:
                fold_scores = Parallel(n_jobs=n_jobs)(delayed(evaluate_fold)(C, train_index, val_index) for train_index, val_index in splits)
            else:
                fold_scores = [evaluate_fold(C, train_index, val_index) for train_index, val_index in splits]
            avg_scores.append(np.mean(fold_scores))

        best_C_index = np.argmax(avg_scores)
        best_C = Cs[best_C_index]

    metrics = {}

    if best_C is not None:
        if penalty == "l1":
            final_model = LogisticRegression(C=best_C, penalty="l1", solver="saga", random_state=seed, max_iter=1000)
        else:
            final_model = LogisticRegression(C=best_C, random_state=seed, max_iter=1000)
    else:
        if penalty == "l1":
            final_model = LogisticRegression(penalty="l1", solver="saga", random_state=seed, max_iter=1000)
        else:
            final_model = LogisticRegression(random_state=seed, max_iter=1000)

    rng = np.random.RandomState(seed)
    shuffle_idx = rng.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
    if best_C is not None:
        metrics["val_auc"] = np.max(avg_scores)
    else:
        metrics["val_auc"] = roc_auc_score(y_train, final_model.predict_proba(X_train)[:, 1])
    if plot:
        plt.semilogx(Cs, avg_scores)
        plt.xlabel("Inverse of Regularization Strength (C)")
        plt.ylabel('auc on validation data')
        plt.title(f'Logistic Regression Performance vs Regularization\nBest C = {best_C:.5f}; auc = {metrics["test_auc"]:.2f}')
        plt.show()
    if return_classifier:
        return metrics, final_model
    return metrics


def find_best_pcareg(X_train, y_train, X_test, y_test, plot=False, max_pca_comps=100):
    # PCA followed by logistic regression, cross-validating over the number of components
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    max_components = min(X_train.shape[0], X_train.shape[1], max_pca_comps)
    pca_dimensions = np.unique(np.logspace(0, np.log2(max_components), num=10, base=2, dtype=int))

    pca = PCA(n_components=max_components)
    X_combined_pca_full = pca.fit_transform(X_combined_scaled)

    best_score = -float('inf')
    best_model = None
    best_n_components = None
    metrics = {}
    if X_combined_pca_full.shape[0] > 3:
        cv = get_cv(X_train)
        scores = []

        for n_components in pca_dimensions:
            X_pca = X_combined_pca_full[:, :n_components]
            fold_scores = []
            splits = get_splits(cv, X_train, y_train)
            for train_index, val_index in splits:
                X_fold_train, X_fold_val = X_pca[train_index], X_pca[val_index]
                y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_fold_train, y_fold_train)
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                fold_scores.append(roc_auc_score(y_fold_val, y_pred_proba))

            avg_score = np.mean(fold_scores)
            scores.append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_model = LogisticRegression(random_state=42, max_iter=1000).fit(X_pca, y_train)
                best_n_components = n_components
                metrics['val_auc'] = best_score
    else:
        best_n_components = X_combined_pca_full.shape[0]
        best_model = LogisticRegression(random_state=42, max_iter=1000).fit(X_combined_pca_full, y_train)
        y_train_pred_proba = best_model.predict_proba(X_combined_pca_full)[:, 1]
        metrics['val_auc'] = roc_auc_score(y_train, y_train_pred_proba)

    X_test_pca = pca.transform(X_test_scaled)[:, :best_n_components]
    y_test_pred = best_model.predict(X_test_pca)

    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    y_test_pred_proba = best_model.predict_proba(X_test_pca)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)

    if plot and X_combined_pca_full.shape[0] > 3:
        plt.semilogx(pca_dimensions, scores)
        plt.xlabel("Number of PCA Components")
        plt.xscale('log', base=2)
        plt.ylabel('auc on validation data')
        plt.title(f'Best PCA dimension: {best_n_components}, auc = {metrics["val_auc"]:.2f}')
        plt.show()

    return metrics


def find_best_knn(X_train, y_train, X_test, y_test, plot=False, n_jobs=-1):
    # k-nearest neighbours with cross-validated k selection
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    metrics = {}
    if X_train.shape[0] > 3:
        cv = get_cv(X_train)
        test_split = get_splits(cv, X_train, y_train)
        min_train_vals = float('inf')
        for split in test_split:
            min_train_vals = min(len(split[0]), min_train_vals)
        max_k = min(X_train.shape[0] - 1, 100, min_train_vals)
        k_values = np.unique(np.logspace(0, np.log2(max_k), num=10, base=2, dtype=int))

        best_score = -float('inf')
        best_model = None
        best_k = None
        scores = []

        def evaluate_fold(k, train_index, val_index):
            X_fold_train, X_fold_val = X_combined_scaled[train_index], X_combined_scaled[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            return roc_auc_score(y_fold_val, y_pred_proba)

        for k in k_values:
            splits = get_splits(cv, X_train, y_train)
            fold_scores = Parallel(n_jobs=n_jobs)(delayed(evaluate_fold)(k, train_index, val_index)
                                                  for train_index, val_index in splits)
            avg_score = np.mean(fold_scores)
            scores.append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_model = KNeighborsClassifier(n_neighbors=k).fit(X_combined_scaled, y_train)
                metrics['val_auc'] = best_score
                best_k = k
    else:
        best_k = 1
        best_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_combined_scaled, y_train)
        y_train_pred_proba = best_model.predict_proba(X_combined_scaled)[:, 1]
        metrics['val_auc'] = roc_auc_score(y_train, y_train_pred_proba)

    y_test_pred = best_model.predict(X_test_scaled)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)

    if plot and X_train.shape[0] > 3:
        plt.semilogx(k_values, scores)
        plt.xlabel("Number of Neighbors (k)")
        plt.xscale('log', base=2)
        plt.ylabel('auc on validation data')
        plt.title(f'Best k: {best_k}, auc = {metrics["val_auc"]:.2f}')
        plt.show()

    return metrics


def find_best_xgboost(X_train, y_train, X_test, y_test, classification=True, binary=True, plot=False, cv_folds=3):
    # random-search XGBoost hyperparameter tuning with cross-validation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if len(X_train) <= 3:
        default_model = xgboost.XGBClassifier()
        default_model.fit(X_train_scaled, y_train)
        y_test_pred = default_model.predict(X_test_scaled)
        metrics = {}
        y_train_proba = default_model.predict_proba(X_train_scaled)[:, 1]
        metrics['val_auc'] = roc_auc_score(y_train, y_train_proba)
        metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
        metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
        y_test_pred_proba = default_model.predict_proba(X_test)[:, 1]
        metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
        return metrics

    param_space = {
        'n_estimators': np.arange(50, 300, step=50),
        'max_depth': np.arange(2, 6),
        'learning_rate': np.logspace(-3, -1, num=10),
        'subsample': np.linspace(0.7, 1.0, num=5),
        'colsample_bytree': np.linspace(0.7, 1.0, num=5),
        'reg_alpha': np.logspace(-3, 1, num=10),
        'reg_lambda': np.logspace(-3, 1, num=10),
        'min_child_weight': np.arange(1, 10)
    }

    cv = get_cv(X_train)
    splits = get_splits(cv, X_train_scaled, y_train)

    best_auc = 0
    best_params = None
    n_iter = 10
    for _ in range(n_iter):
        params = {k: random.choice(v) for k, v in param_space.items()}
        model = xgboost.XGBClassifier(**params, eval_metric='logloss')
        cv_scores = []
        for train_idx, val_idx in splits:
            model.fit(X_train_scaled[train_idx], y_train[train_idx])
            y_fold_val_proba = model.predict_proba(X_train_scaled[val_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_train[val_idx], y_fold_val_proba))
        mean_auc = np.mean(cv_scores)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    metrics = {}
    best_model = xgboost.XGBClassifier(**best_params, eval_metric='logloss')
    best_model.fit(X_train_scaled, y_train)
    y_test_pred = best_model.predict(X_test_scaled)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
    metrics['val_auc'] = best_auc

    return metrics


def find_best_mlp(X_train, y_train, X_test, y_test, classification=True, binary=True, plot=False):
    # random-search MLP hyperparameter tuning with cross-validation
    X_combined = X_train
    y_combined = y_train

    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined)
    X_test_scaled = scaler.transform(X_test)

    metrics = {}
    if X_train.shape[0] <= 3:
        best_model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
        best_model.fit(X_combined_scaled, y_combined)
        y_train_pred_proba = best_model.predict_proba(X_combined_scaled)[:, 1]
        metrics['val_auc'] = roc_auc_score(y_train, y_train_pred_proba)
    else:
        param_dist = {
            'hidden_layer_sizes': [(16,), (32,), (64,), (16, 16), (32, 32), (64, 64), (16, 16, 16), (32, 32, 32), (64, 64, 64)],
            'learning_rate_init': np.logspace(-4, -2, num=5),
            'alpha': np.logspace(-5, -2, num=5),
            'activation': ['relu'],
            'solver': ['adam'],
        }

        cv = get_cv(X_train)
        splits = get_splits(cv, X_combined_scaled, y_combined)

        best_score = -float('inf')
        best_params = None
        best_model = None

        n_iter = 1
        np.random.seed(42)

        for _ in range(n_iter):
            curr_params = {
                'hidden_layer_sizes': param_dist['hidden_layer_sizes'][np.random.randint(len(param_dist['hidden_layer_sizes']))],
                'learning_rate_init': param_dist['learning_rate_init'][np.random.randint(len(param_dist['learning_rate_init']))],
                'alpha': np.random.choice(param_dist['alpha']),
                'activation': 'relu',
                'solver': 'adam'
            }

            cv_scores = []
            for train_idx, val_idx in splits:
                model = MLPClassifier(max_iter=1000, random_state=42, **curr_params)
                model.fit(X_combined_scaled[train_idx], y_combined[train_idx])
                val_proba = model.predict_proba(X_combined_scaled[val_idx])[:, 1]
                cv_scores.append(roc_auc_score(y_combined[val_idx], val_proba))

            mean_cv_score = np.mean(cv_scores)
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_params = curr_params

        metrics['val_auc'] = best_score
        best_model = MLPClassifier(max_iter=1000, random_state=42, **best_params)
        best_model.fit(X_combined_scaled, y_combined)

    y_test_pred = best_model.predict(X_test_scaled)
    metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted')
    metrics['test_acc'] = accuracy_score(y_test, y_test_pred)
    y_test_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    metrics['test_auc'] = roc_auc_score(y_test, y_test_pred_proba)
    return metrics
