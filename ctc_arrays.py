'''
For two 1-d arrays
find the arr2 indicies for the corresponding arr1 input
'''
import numpy as np

def awhere(arr1,arr2):
	'''
	This only works if arr1 is a subset of arr2
	'''
	#sort arr1
	id_sort = np.argsort(arr2)
	sorted_arr2 = arr2[id_sort]
	sorted_id = np.searchsorted(sorted_arr2, arr1) 
	sorted_id[arr2[sorted_id] != arr1] = -1
	return sorted_id
