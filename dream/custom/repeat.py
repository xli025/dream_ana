import numpy as np

def repeat(arr, n_arr):
    return np.repeat(arr, n_arr.astype(int))

def repeat_dest4_280(arr, n_arr, dest, t280, num1, num2):
    inds = (dest==num1)&(t280==num2)
    arr[inds] = np.nan        
    return repeat(arr, n_arr)
