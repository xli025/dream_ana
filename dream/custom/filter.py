import numpy as np
#
def dest4_280(arr, dest, t280, num1, num2):
    inds = (dest==num1)&(t280==num2)
    arr[inds] = np.nan
    return arr

def filter_dest(arr, dest, num):
    return arr[dest==num]

def duck_goose_arr(arr, n_arr, ec, ec_01):
    ec_repeat = np.repeat(ec, n_arr.astype(int))
    inds = ec_repeat == ec_01
    return arr[inds]


def duck_goose_arr1(arr, ec, ec_01):
    inds = ec == ec_01
    return arr[inds]

def duck_goose_arr_gatedOn_xy(arr, n_arr, ec, arr1, arr2, ec_01, l1, r1, l2, r2):
    inds1 = (arr1>l1)&(arr1<r1)&(arr2>l2)&(arr2<r2)
    ec_repeat = np.repeat(ec, n_arr.astype(int))
    inds2 = ec_repeat == ec_01
    inds = inds1&inds2
    return arr[inds]
    

def atm(line, xgmd, ec, xgmd_min, ec_01):
    if xgmd[-1]>xgmd_min and ec[-1]==ec_01:
        return line


def gate1D_count(arr, l, r):
    inds = (arr>l)&(arr<r)
    return inds.sum()


def a_gatedOn_b(arr1, arr2, l2, r2):
    inds = (arr2>l2)&(arr2<r2)
    return arr1[inds]


def a_gatedOn_bc(arr1, arr2, arr3, l2, r2, l3, r3):
    inds = (arr2>l2)&(arr2<r2)&(arr3>l3)&(arr3<r3)
    return arr1[inds]


def n_gatedOn_abc(n_arr, arr1, arr2, arr3, l1, r1, l2, r2, l3, r3):
    inds = (arr1>l1)&(arr1<r1)&(arr2>l2)&(arr2<r2)&(arr3>l3)&(arr3<r3)
    chunk_ids = np.repeat(np.arange(n_arr.size), n_arr.astype(int))       
    n_arr = np.bincount(chunk_ids, weights=inds,
                        minlength=n_arr.size)

    return n_arr.astype(int)
