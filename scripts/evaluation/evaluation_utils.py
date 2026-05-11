# Import packages
import numpy as np

# Code used for the single_deletion and single_insertion experiments

def select_concepts(t, X, decomposition, idx):
    """
    Select component at the given index in decomposition from the given data.
    Used in single_insertion.py.
    NOTE: this function only works if terms are ordered based on the calculation
    order, where terms of the lowest rank are put first
    t: timestamps (are not currently used, just make sure its the same length as X)
    X: data to select concept (component) from
    decomposition: CustomDecomposition, EMD or DWT to apply
    idx: component at index idx is selected from X
    returns: selected data
    """ 
    # Select component indices to remove (all but idx)
    delete = [i for i in range(len(decomposition.component_names)) if i != idx]
    return filter_concepts(t, X, decomposition, delete)

def delete_concept(t, X, decomposition, idx):
    """
    Filter component at the given index in decomposition from the given data.
    Used in single_deletion.py.
    NOTE: this function only works if terms are ordered based on the calculation
    order, where terms of the lowest rank are put first
    t: timestamps (are not currently used, just make sure its the same length as X)
    X: data to remove concept (component) from
    decomposition: CustomDecomposition, EMD or DWT to apply
    idx: component at index idx is removed from X
    returns: filtered data
    """
    # Select component indices to remove (only idx)
    delete = [idx]
    return filter_concepts(t, X, decomposition, delete)

def filter_concepts(t, X, decomposition, delete):
    """
    Filter components from X at the given indices in delete.
    Used in single_deletion.py.
    NOTE: this function only works if terms are ordered based on the calculation
    order, where terms of the lowest rank are put first
    t: timestamps (are not currently used, just make sure its the same length as X)
    X: data to remove concept (component) from
    decomposition: CustomDecomposition, EMD or DWT to apply
    delete: components at indices in delete are removed from X
    returns: filtered data
    """
    print('components removed:', delete)

    components = decomposition.decompose(t, X)         # Decompose the given data
    n_components = components.shape[-1]             # Determine the number of components

    # Only consider multiplicative and additive operators
    operators_filtered = [op for op in decomposition.operators if op in ['*', '+']]

    # Copy components object
    components_filtered = np.array(components)

    # First set additive components that come before an additive operator to 0
    for i, opp in enumerate(operators_filtered):
        if opp == '+':
            if i in delete:
                components_filtered[:, :, i] = np.zeros((components.shape[0], components.shape[1]))

    # Then consider multiplicative components
    for i, opp in enumerate(operators_filtered):
        if opp == '*':
            # If all components after the currently selected component should be removed,
            # but this one should not set the next one to 1 and the ones after to 0
            if i not in delete and all([n in delete for n in range(i + 1, n_components)]):
                components_filtered[:, :, i + 1] = np.ones((components.shape[0], components.shape[1]))
                if i + 2 <= n_components:
                    for n in range(i + 2, n_components):
                        components_filtered[:, :, n] = np.zeros((components.shape[0], components.shape[1]))
            # If this component should be removed and all that follow as well, set all of them to zero
            elif i in delete and all([n in delete for n in range(i + 1, n_components)]):
                for n in range(i, n_components):
                    components_filtered[:, :, n] = np.zeros((components.shape[0], components.shape[1]))
            # Otherwise if the current component should be removed set it to 1
            elif i in delete:
                components_filtered[:, :, i] = np.ones((components.shape[0], components.shape[1]))

    # Compose data from the filtered components
    x_filtered = decomposition.compose(components_filtered)
    print(x_filtered.shape)
    return x_filtered



# def select_concept(t, X, decomposition, idx):
#     """
#     Select component at the given index in decomposition from the given data.
#     Used in single_insertion.py.
#     NOTE: this function only works if terms are ordered based on the calculation
#     order, where terms of the lowest rank are put first
#     X: data to select concept (component) from
#     decomposition: CustomDecomposition, EMD or DWT to apply
#     idx: component at index idx is selected from X
#     returns: selected data
#     """
#     components = decomposition.decompose(X)         # Decompose the given data
#     n_components = components.shape[-1]             # Determine the number of components

#     # Select component indices to remove (all but idx)
#     delete = [i for i in range(len(decomposition.component_names)) if i != idx]
#     print('components removed:', delete)

#     # Only consider multiplicative and additive operators
#     operators_filtered = [op for op in decomposition.operators if op in ['*', '+']]

#     # Copy components object
#     components_filtered = np.array(components)

#     # First set additive components that come before an additive operator to 0
#     for i, opp in enumerate(operators_filtered):
#         if opp == '+':
#             if i in delete:
#                 components_filtered[:, :, i] = np.zeros((components.shape[0], components.shape[1]))

#     # Then consider multiplicative components
#     for i, opp in enumerate(operators_filtered):
#         if opp == '*':
#             # If all components after the currently selected component should be removed,
#             # but this one should not set the next one to 1 and the ones after to 0
#             if i not in delete and all([n in delete for n in range(i + 1, n_components)]):
#                 components_filtered[:, :, i + 1] = np.ones((components.shape[0], components.shape[1]))
#                 if i + 2 <= n_components:
#                     for n in range(i + 2, n_components):
#                         components_filtered[:, :, n] = np.zeros((components.shape[0], components.shape[1]))
#             # If this component should be removed and all that follow as well, set all of them to zero
#             elif i in delete and all([n in delete for n in range(i + 1, n_components)]):
#                 for n in range(i, n_components):
#                     components_filtered[:, :, n] = np.zeros((components.shape[0], components.shape[1]))
#             # Otherwise if the current component should be removed set it to 1
#             elif i in delete:
#                 components_filtered[:, :, i] = np.ones((components.shape[0], components.shape[1]))

#     # Compose data from the filtered components
#     x_filtered = decomposition.compose(components_filtered[:, :, None, :])
#     return x_filtered

# def delete_concept(t, X, decomposition, idx):
#     """
#     Filter component at the given index in decomposition from the given data.
#     Used in single_deletion.py.
#     NOTE: this function only works if terms are ordered based on the calculation
#     order, where terms of the lowest rank are put first
#     X: data to remove concept (component) from
#     decomposition: CustomDecomposition, EMD or DWT to apply
#     idx: component at index idx is removed from X
#     returns: filtered data
#     """
#     components = decomposition.decompose(X)         # Decompose the given data
#     n_components = components.shape[-1]             # Determine the number of components

#     # Select component indices to remove only idx
#     delete = [idx]
#     operators_filtered = [op for op in decomposition.operators if op in ['*', '+']]

#     components_filtered = np.array(components)
#     print('components removed:', delete)

#     for i, opp in enumerate(operators_filtered):
#         if opp == '+':
#             if i in delete:
#                 components_filtered[:, :, i] = np.zeros((components.shape[0], components.shape[1]))

#     for i, opp in enumerate(operators_filtered):
#         if opp == '*':
#             if i not in delete and all([n in delete for n in range(i + 1, n_components)]):
#                 components_filtered[:, :, i + 1] = np.ones((components.shape[0], components.shape[1]))
#                 if i + 2 <= n_components:
#                     for n in range(i + 2, n_components):
#                         components_filtered[:, :, n] = np.zeros((components.shape[0], components.shape[1]))
#             elif i in delete and all([n in delete for n in range(i + 1, n_components)]):
#                 for n in range(i, n_components):
#                     components_filtered[:, :, n] = np.zeros((components.shape[0], components.shape[1]))
#             elif i in delete:
#                 components_filtered[:, :, i] = np.ones((components.shape[0], components.shape[1]))

#     x_filtered = decomposition.compose(components_filtered[:, :, None, :])
#     return x_filtered