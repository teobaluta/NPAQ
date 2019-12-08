import os
import sys
import utils
import definitions
import numpy as np

def split_shape_str(shape_str):
	temp = shape_str.split(',')
	return [int(temp[0]), int(temp[1])]

def gen_mask(mask_shape, row_loc, col_loc, dataset):
	shape_list = split_shape_str(mask_shape)
	init_mask = np.zeros((shape_list[0], shape_list[1]))

	row_list = split_shape_str(row_loc)
	col_list = split_shape_str(col_loc)

	row_index = []
	col_num = col_list[1] - col_list[0]
	for row_id in range(row_list[0], row_list[1]):
		row_index += [row_id] * col_num
	row_index = np.asarray(row_index)

	row_num = row_list[1] - row_list[0]
	col_index = range(col_list[0], col_list[1]) * row_num
	col_index = np.asarray(col_index)

	one_list = [1]*(row_num*col_num)
	one_list = np.asarray(one_list)
	init_mask[row_index, col_index] = one_list

	init_mask = init_mask.flatten().astype(np.int)
	mask_path = os.path.join(definitions.TROJAN_DIR, 'mask_%s_%d.bin' % (
		dataset, shape_list[0]*shape_list[1]))
	utils.write_sample(mask_path, list(init_mask))


if __name__ == '__main__':
	mask_shape = sys.argv[1]
	row_loc = sys.argv[2]
	col_loc = sys.argv[3]
	dataset = sys.argv[4]
	gen_mask(mask_shape, row_loc, col_loc, dataset)
