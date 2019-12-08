import pickle
from copy import copy
import numpy as np
import sys

def read_pkl(file_path):
    with open(file_path) as f:
        info = pickle.load(f)
    return info

def test_clip_grad(grad_dict, grad_bound, clip_grad):
    success_flag = True
    for name in grad_dict:
        current_grad = copy(np.asarray(grad_dict[name]))
        processed_grad = []
        for item in current_grad:
            clip_value = np.linalg.norm(item)/grad_bound
            div_value = max(1.0, clip_value)
            processed_grad.append(item/div_value)
        processed_grad = np.asarray(processed_grad)
        test_clip_grad = processed_grad.sum(axis=0)
        if np.array_equal(clip_grad[name], test_clip_grad):
            pass
        else:
            index = np.where(clip_grad[name] != test_clip_grad)
            print(index)
            print clip_grad[name]
            print test_clip_grad
            success_flag = False
            return success_flag
    return success_flag

def test_add_noise(noise_dict, clip_grad, final_grad, group_size):
    success_flag = True
    for name in clip_grad:
        noise = np.asarray(noise_dict[name])
        grad = clip_grad[name]
        processed_grad = (grad + noise) / group_size
        if np.array_equal(processed_grad, final_grad[name]):
            pass
        else:
            success_flag = False
            return success_flag
    return success_flag

def test_update_grad(grad, lr, origin_weight, updated_weight):
    success_flag = True
    for name in grad:
        process_weight = origin_weight[name] - grad[name]*lr
        if np.array_equal(process_weight, updated_weight[name]):
            pass
        else:
            success_flag = False
            return success_flag
    return success_flag


if __name__ == '__main__':
    file_path = sys.argv[1]
    grad_bound = float(sys.argv[2])
    group_size = int(sys.argv[3])
    lr = float(sys.argv[4])
    info = read_pkl(file_path)
    if test_clip_grad(info['grad'], grad_bound, info['cliped_grad']):
        print('Succeed in test_clip_grad!')
    else:
        print('Failed in test_clip_grad!')

    if test_add_noise(info['noise'], info['cliped_grad'], info['final_grad'], group_size):
        print('Succeed in test_add_noise!')
    else:
        print('Failed in test_add_noise!')

    if test_update_grad(info['final_grad'], lr, info['origin_weight'], info['updated_weight']):
        print('Succeed in test_update_grad!')
    else:
        print('Failed in test_update_grad!')
