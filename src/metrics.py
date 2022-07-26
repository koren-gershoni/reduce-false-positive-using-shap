# def mrr(arr, k):
#     pass

def precision_k(arr, k):
    values, indices = zip(*sorted(zip(arr, range(len(arr))), reverse=True))
    values, indices = values[:k], indices[:k]
    return values, indices