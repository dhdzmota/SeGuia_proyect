
def transform_numerical_drought_index(idx):
    """This function transforms an index on the DROUGHT_INDEX into
    the respective numerical form.

    For example:
    none -> 0
    D1 -> 1
    D2 -> 2
    ...
    DN-> N
    Where N is an integer.
    """
    if isinstance(idx, str):
        return float(idx.replace('D', ''))
    return 0
