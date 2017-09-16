import numpy as np

# data
# weights
# strides
# padding(opt)

# def get_patches():
# 	pass

"""
data : 3d [in_height 4, in_width 4, in_depth 64]
//weights : 5d [in_height (4), in_width (4), in_depth (64), kernel*kernel (3x3), depth (64)]
weights : [kernel, kernel, in_depth, in_height-diff, in_width-diff, depth]
biases : ?
"""
def LC_layer(data, weights, strides, kernel_size=3):#, padding="SAME"):		
	shape = list(weights.shape)
	filters = shape[5]
	Y = shape[3]
	X = shape[4]
	#shape = list(data.shape)
	#print shape
	diff = kernel_size - 1
	sumb = np.ndarray(shape=(Y, X, filters), dtype=np.float32)
	for d in range(filters):
		for h in range(Y, strides[1]):
			for w in range(X, strides[0]):
				sumb[h, w, d] = np.sum(data[h:h+kernel_size, w:w+kernel_size, :] * weights[:, :, :, h:h+kernel_size, w:w+kernel_size, d])

	print sumb.shape

data = np.ndarray(shape=(4, 4, 64), dtype=np.float32)
weights = np.ndarray(shape=(3, 3, 64, 2, 2, 5), dtype=np.float32)
LC_layer(data, weights, [2,2])