
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.decomposition import PCA



class Auto_cluster:

    def __init__(self, estimators=None):
        self.estimators = estimators
         
        pass
    
    def fit_predict(self, X, y=None, decomposer={'name':PCA, 'args':[], 'kwargs': {'n_components':2}}):
        """
        fit_predict will train given estimator and predict cluster membership for each sample
        """
        shape = X.shape
        df_type = isinstance(X, pd.core.frame.DataFrame)

        if df_type:
            column_names = X.columns
            index = X.index

        if decomposer is not None:
            X = decomposer['name'](*decomposer['args'], **decomposer['kwargs']).fit_transform(X)

            if df_type:
                if decomposer['name'].__name__ == 'PCA':
                    X = pd.DataFrame(X, index=index, columns=['component_' + str(i + 1) for i in
                    range(decomposer['kwargs']['n_components'])])

                else:
                    X = pd.DataFrame(X, index=index, columns=['component_1', 'component_2'])

            # if decomposition is applied, then n_components will be set accordingly in hyperparameter configuration

            for estimator in self.estimators:
                if 'n_clusters' in estimator['kwargs'].keys():
                    if decomposer['name'].__name__ == 'PCA':
                        estimator['kwargs']['n_clusters'] = decomposer['kwargs']['n_components']
                    else:
                        estimator['kwargs']['n_clusters'] = 2

        # This dictionary will hold predictions for each estimator
        predictions = []
        performance_metrics = {}

        for estimator in self.estimators:
            labels = estimator['estimator'](*estimator['args'], **estimator['kwargs']).fit_predict(X)
            #print('labels.........', labels)
            estimator['estimator'].n_clusters_ = len(np.unique(labels)) # getting number of clusters
            metrics = self._get_cluster_metrics(estimator['estimator'].__name__, estimator['estimator'].n_clusters_, X, labels, y)
            predictions.append({estimator['estimator'].__name__: labels})
            performance_metrics[estimator['estimator'].__name__] = metrics
            
        self.predictions = predictions
        self.performance_metrics = performance_metrics

        return predictions, performance_metrics
    
    # Printing cluster metrics for given arguments
    def _get_cluster_metrics(self, name, n_clusters_, X, pred_labels, true_labels=None):
        from sklearn.metrics import homogeneity_score, \
            completeness_score, \
            v_measure_score, \
            adjusted_rand_score, \
            adjusted_mutual_info_score, \
            silhouette_score, \
            davies_bouldin_score, \
            calinski_harabasz_score    

        print("""-------------- %s metrics --------------""" % name)
        if len(np.unique(pred_labels)) >= 2:

            silh_co = silhouette_score(X, pred_labels)
            davies_co = davies_bouldin_score(X, pred_labels)
            calinski_co = calinski_harabasz_score (X, pred_labels)

            # if true_labels is not None:

            #     h_score = homogeneity_score(true_labels, pred_labels)
            #     c_score = completeness_score(true_labels, pred_labels)
            #     vm_score = v_measure_score(true_labels, pred_labels)
            #     adj_r_score = adjusted_rand_score(true_labels, pred_labels)
            #     adj_mut_info_score = adjusted_mutual_info_score(true_labels, pred_labels)

            #     metrics = {"Silhouette Coefficient": silh_co,
            #                "Estimated number of clusters": n_clusters_,
            #                "Homogeneity": h_score,
            #                "Completeness": c_score,
            #                "V-measure": vm_score,
            #                "Adjusted Rand Index": adj_r_score,
            #                "Adjusted Mutual Information": adj_mut_info_score}

            #     for k, v in metrics.items():
            #         print("\t%s: %0.3f" % (k, v))

            #     return metrics

            metrics = {"Silhouette Coefficient": silh_co,
                         "Davies Coefficient": davies_co,
                         "Calinski Coefficient": calinski_co,
                        "Estimated number of clusters": n_clusters_}

            for k, v in metrics.items():
                print("\t%s: %0.3f" % (k, v))

            return metrics

        else:
            print("\t# of predicted labels is {}, can not produce metrics. \n".format(np.unique(pred_labels)))
            

