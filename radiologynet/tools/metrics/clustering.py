from radiologynet.logging import log
import pandas as pd
import numpy as np
import typing


def cluster_and_get_metrics(
    algorithm: str,
    train_data: np.ndarray,
    test_data: np.ndarray,
    original_df: pd.DataFrame,
    list_n_clusters: typing.List[int],
    verbose: bool = False,
    random_state: int = 1,
    clusterer_save_dir: str = None
) -> pd.DataFrame:
    """Cluster data and calculate homogeneity and mutual
    information of obtained clusters.

    Args:
        algorithm (str): Which algorithm to use.
        train_data: (np.ndarray): Data to fit clusters on.
        test_data: (np.ndarray): Data to use for validation.
            The trained algorithm will predict classes in this dataset,
            and then metrics will be calculated based on clustering results.
        original_df (pd.DataFrame): Dataframe containing original,
            unencoded data.
        list_n_clusters (int): A list of Number of clusters to test.
            Multiple clusterings will be performed to test
            each cluster count.
        verbose (bool, optional): Print useful logs.
            Defaults to False.
        random_state (int, optional): For reproducibilty.
            Defaults to 1.
        clusterer_save_dir (str, optional): Save cluster models
            to this dir. If None, nothing will be saved.

    Returns:
        pd.DataFrame: Dataframe containing the results.
    """

    assert (algorithm in ['k-means', 'k-medoids'],
            'Algorithm not supported.')

    # distance and method to use for clustering
    # (only applicable to kmedoids)
    dist = 'manhattan'
    method = 'alternate'
    max_iter: int = 300
    initalizer = f'{algorithm}++'  # k-medoids++ or k-means++
    from sklearn.cluster import KMeans
    from sklearn_extra.cluster import KMedoids
    from sklearn.metrics import homogeneity_score,\
        normalized_mutual_info_score,\
        completeness_score

    _cols_for_stats = ['Modality', 'BodyPartExamined']
    _statnames = ['NMI', 'HS', 'CS']
    _tmp = [
        'Algorithm',
        'NClusters',
        'Distance',
        'Initializer',
        'MaxIter',
        *[f'{_s}{_c}' for _s in _statnames for _c in _cols_for_stats]
    ]

    results = pd.DataFrame(columns=_tmp)
    for n_clusters in list_n_clusters:
        if algorithm == 'k-means':
            clusterer = KMeans(
                n_clusters=n_clusters,
                init=initalizer,
                random_state=random_state,
                algorithm='elkan'
            )
            log(
                f'Initialised clustering using {algorithm},' +
                f' n_clusters={n_clusters},' +
                f' max_iter={max_iter}',
                verbose=verbose
            )
            dist = '-'
        else:
            clusterer = KMedoids(
                n_clusters=n_clusters,
                metric=dist,
                method=method,
                init=initalizer,
                random_state=random_state,
                max_iter=max_iter
            )
            log(
                f'Initialised clustering using {algorithm},' +
                f' n_clusters={n_clusters},' +
                f' method={method},' +
                f' max_iter={max_iter},' +
                f' dist={dist}',
                verbose=verbose
            )
        clusterer = clusterer.fit(train_data)
        log(
            'Successfully clustered & predicted classes.' +
            ' Calculating metrics...',
            verbose=verbose
        )
        # calculate homogeneity and NMI
        # add new row to results df, filled with zeros
        _idx = len(results)
        results.loc[_idx] = 0
        results.at[_idx, 'Algorithm'] = algorithm
        results.at[_idx, 'NClusters'] = n_clusters
        results.at[_idx, 'Distance'] = dist
        results.at[_idx, 'MaxIter'] = max_iter
        results.at[_idx, 'Initializer'] = initalizer

        for attr in _cols_for_stats:
            true_labels = original_df[attr].to_numpy()
            assigned_clusters = clusterer.predict(test_data)
            for stat in _statnames:
                metric = 0
                if stat == 'NMI':
                    metric = normalized_mutual_info_score(
                        true_labels, assigned_clusters)
                elif stat == 'HS':
                    metric = homogeneity_score(
                        true_labels, assigned_clusters)
                elif stat == 'CS':
                    metric = completeness_score(
                        true_labels, assigned_clusters)
                results.at[_idx, f'{stat}{attr}'] = metric
            del assigned_clusters, true_labels
        # save the cluster model before deleting
        if clusterer_save_dir is not None:
            import os
            import pickle
            os.makedirs(clusterer_save_dir, exist_ok=True)
            _path = os.path.join(clusterer_save_dir,
                                 f'{algorithm}-n_clusters-{n_clusters}.pkl')
            log(f'Saving cluster model to {_path}', verbose=verbose)
            with open(_path, 'wb') as file:
                pickle.dump(clusterer, file)

        del clusterer

    log('Done calculating metrics. Returning result...', verbose=verbose)
    return results
