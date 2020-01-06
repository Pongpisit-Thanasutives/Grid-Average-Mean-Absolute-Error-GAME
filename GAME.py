import numpy as np

def padding(array, values, axis):
    # This function should be doing post padding 0s.
    if axis not in {0,1}:
        print("Error! axis should be 0 or 1.")
        
    dim = array.shape
    new_dim = [0,0]
    for i in range(2):
        if i == axis:
            new_dim[i] = dim[i]+1
        else:
            new_dim[i] = dim[i]
    new_dim = tuple(new_dim)
    new_array = np.zeros(new_dim)
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            new_array[i][j] = array[i][j]
    return new_array

def adjust_dim(array):
    # Make the dim even
    if array.shape[0]%2 != 0:
        array = padding(array, 0, 0)
    if array.shape[1]%2 != 0:
        array = padding(array, 0, 1)
    return array

def GAME_recursive(density, gt, currentLevel, targetLevel):
    if currentLevel == targetLevel:
        game = abs(np.sum(density) - np.sum(gt))
        return np.round(game, 3)
    
    else:
        density = adjust_dim(density)
        gt = adjust_dim(gt)
        density_slice = []; gt_slice = []
        
        density_slice.append(density[0:density.shape[0]//2, 0:density.shape[1]//2])
        density_slice.append(density[0:density.shape[0]//2, density.shape[1]//2:])
        density_slice.append(density[density.shape[0]//2:, 0:density.shape[1]//2])
        density_slice.append(density[density.shape[0]//2:, density.shape[1]//2:])

        gt_slice.append(gt[0:gt.shape[0]//2, 0:gt.shape[1]//2])
        gt_slice.append(gt[0:gt.shape[0]//2, gt.shape[1]//2:])
        gt_slice.append(gt[gt.shape[0]//2:, 0:gt.shape[1]//2])
        gt_slice.append(gt[gt.shape[0]//2:, gt.shape[1]//2:])
        
        currentLevel = currentLevel +1;
        res = []
        for a in range(4):
            res.append(GAME_recursive(density_slice[a], gt_slice[a], currentLevel, targetLevel))
        game = sum(res)
        return np.round(game, 3)

def GAME_metric(preds, gts, l):
	res = []
	for i in range(len(gts)):
		res.append(GAME_recursive(preds[i], gts[i], 0, l))
	return np.mean(res)