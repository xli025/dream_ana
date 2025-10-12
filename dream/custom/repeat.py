import numpy as np

def repeat(arr, n_arr):
        #print(np.unique(arr))
        return np.repeat(arr*1e12, n_arr.astype(int))


def repeat_a_count_gatedOn_abc(arr, n_arr, arr1, arr2, arr3, l1, r1, l2, r2, l3, r3):
    inds = (arr1>l1)&(arr1<r1)&(arr2>l2)&(arr2<r2)&(arr3>l3)&(arr3<r3)
    print('shape1:',arr.shape, n_arr.shape)
    print('arr, n_arr:', arr, n_arr)
    n_arr = n_arr.astype(int)
    chunk_ids = np.repeat(np.arange(len(n_arr)), n_arr)     # [0,0,1,1,1,2]
    n_arr = np.bincount(chunk_ids, weights=(inds))
    print('shape2:',arr.shape, n_arr.shape, chunk_ids.shape, inds.shape)
    print('arr, chunk_ids, n_arr:', arr, chunk_ids, n_arr)
    if n_arr.size==0: return None
    return np.repeat(arr*1e12, n_arr.astype(int))