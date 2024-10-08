import os
import json
import numpy as np
import pandas as pd
from os.path import join

def read_csv_from_dir(path, prefix: str = None, suffix: str = None, shape: tuple = None, dtype = None):
    files = []

    # 遍历目录，找到所有符合"J_*.csv"格式的文件
    for filename in os.listdir(path):
        if filename.startswith(prefix) and filename.endswith(suffix):
            files.append(filename)
    
    # 对文件名进行排序，确保按数字顺序（假设文件名格式为 'J_数字.csv'）
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # 创建一个空列表来存储所有的numpy数组
    arrays = []

    # 按顺序读取每个文件
    for filename in files:
        full_path = os.path.join(path, filename)
        # 读取CSV文件
        df = pd.read_csv(full_path, header=None)
        # 将DataFrame转换为numpy数组并添加到列表中
        if shape is not None:
            df = df.to_numpy(dtype=dtype).reshape(shape)
        else:
            df = df.to_numpy(dtype=dtype).reshape(-1)
        arrays.append(df)

    return arrays

def load_test_data(data_args, num_measurements: int = None):
    if num_measurements == None:
        num_measurements = 1000
    elif num_measurements > 1000:
        raise ValueError("The number of measurements should be less than 1000.")
    
    path = os.path.join('../DiffuSeq/data/', data_args.data_dir)
    if not os.path.exists(path):
        raise ValueError(f"Data directory {path} does not exist.")
    
    print('#'*30, '\nLoading test dataset from {}...'.format(data_args.data_dir))

    dataset = {}

    conditions = read_csv_from_dir(join(path, 'J'), 
                                   prefix='J_', 
                                   suffix='.csv', 
                                   dtype=np.float32)
    conditions_array = np.stack(conditions, axis=0) # shape is (n_samples, dim_conditions)


    bits = read_csv_from_dir(join(path, 'samples'), 
                             prefix='povm_samples_', 
                             suffix='.csv', 
                             shape=(num_measurements, -1),
                             dtype=np.int64)
    recipes = read_csv_from_dir(join(path, 'basis'),
                                prefix='povm_basis_',
                                suffix='.csv',
                                shape=(num_measurements, -1),
                                dtype=np.int64)
    assert len(bits) == len(recipes) == len(conditions_array), 'The number of bits and recipes should be the same.'
    # bits_array = np.stack(bits, axis=0)
    bits_array = np.stack(bits, axis=0) - 1
    recipes_array = np.stack(recipes, axis=0)
    measurements = 2 * recipes_array + bits_array # shape is (n_samples, n_measurements, n_qubits)
    assert measurements.ndim == 3, 'The shape of measurements should be (n_samples, n_measurements, n_qubits).'
    
    # expand the conditions array to the same shape as measurements
    # _shape = (measurements.shape[0], measurements.shape[1], conditions_array.shape[-1])
    # conditions_array = np.broadcast_to(conditions_array[:, np.newaxis, :], _shape)

    # reshape to (n_samples * n_measurements, -1)
    dataset['conditions'] = conditions_array # shape is (n_samples, dim_conditions)
    dataset['input_ids'] = measurements # shape is (n_samples, n_measurements, n_qubits)

    # dataset['input_mask'] = np.ones((dataset['input_ids'].shape[0], dataset['input_ids'].shape[1] + 1), dtype=np.int64) # (n_samples * n_measurements, 1 + n_qubits)
    # dataset['input_mask'][:, 0] = 0 # set the first element to 0
    # assert dataset['conditions'].shape[0] == dataset['input_ids'].shape[0] == dataset['input_mask'].shape[0], 'The number of conditions and measurements should be the same.'
    # assert dataset['input_ids'].ndim == dataset['input_mask'].ndim and dataset['input_mask'].shape[-1] == dataset['input_ids'].shape[-1] + 1 , 'The shape of input_ids and input_mask should be the same.'
    
    print('### Total number of samples:', dataset['input_ids'].shape[0])
    print('### The dimension of conditions:', dataset['conditions'].shape[-1])
    print('### The number of qubits:', dataset['input_ids'].shape[-1])
    print('### Data samples...\n', dataset['conditions'][:2], '\n', dataset['input_ids'][0][:2])
        
    return dataset


        




def load_data(data_args, split='train'):
    print('#'*30, '\nLoading dataset from {}...'.format(data_args.data_dir))


    dataset = {}
    
    # if split == 'train':
    #     print('### Loading form the TRAIN set...')
    #     path = f'{data_args.data_dir}/train.jsonl'
    # elif split == 'valid':
    #     print('### Loading form the VALID set...')
    #     path = f'{data_args.data_dir}/valid.jsonl'
    # elif split == 'test':
    #     print('### Loading form the TEST set...')
    #     path = f'{data_args.data_dir}/test.jsonl'
    # else:
    #     assert False, "invalid split for dataset"
    path = os.path.join('../DiffuSeq/data/', data_args.data_dir)
    if not os.path.exists(path):
        raise ValueError(f"Data directory {path} does not exist.")

    num_measurements = data_args.num_measurements
    num_qubits = data_args.num_qubits

    conditions = read_csv_from_dir(join(path, 'J'), 
                                   prefix='J_', 
                                   suffix='.csv', 
                                   dtype=np.float32)
    conditions_array = np.stack(conditions, axis=0) # shape is (n_samples, dim_conditions)

    if split == 'test':
        dataset['conditions'] = conditions_array
        return dataset



    bits = read_csv_from_dir(join(path, 'samples'), 
                             prefix='povm_samples_', 
                             suffix='.csv', 
                             shape=(num_measurements, -1),
                             dtype=np.int64)
    recipes = read_csv_from_dir(join(path, 'basis'),
                                prefix='povm_basis_',
                                suffix='.csv',
                                shape=(num_measurements, -1),
                                dtype=np.int64)
    assert len(bits) == len(recipes) == len(conditions_array), 'The number of bits and recipes should be the same.'
    # bits_array = np.stack(bits, axis=0)
    bits_array = np.stack(bits, axis=0) - 1
    recipes_array = np.stack(recipes, axis=0)
    measurements = 2 * recipes_array + bits_array # shape is (n_samples, n_measurements, n_qubits)
    assert measurements.ndim == 3, 'The shape of measurements should be (n_samples, n_measurements, n_qubits).'
    
    for i in range(len(conditions_array)):
        k = tuple(conditions_array[i])
        assert k not in dataset, 'The conditions should be unique.'
        v = measurements[i]
        dataset[k] = v
    
    print('### Total number of samples:', len(conditions_array))
    print('### The dimension of conditions:', conditions_array.shape[-1])
    print('### The dimension of measurements:', measurements.shape[-1])
    # print('### Data samples...\n', dataset['conditions'][:2], '\n', dataset['input_ids'][:2])
        
    return dataset