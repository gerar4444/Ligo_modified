import torch
import numpy as np

normal_model_file = 'output_baseline_6L_768H/checkpoints/checkpoint-00000000/pytorch_model.bin'
normal_model_copy = torch.load(normal_model_file, map_location=torch.device('cpu'))

normal_model_copy['bert.embeddings.word_embeddings.weight'] = small_model_coeff['bert.embeddings.word_embeddings.weight']
normal_model_copy['bert.embeddings.position_embeddings.weight'] = small_model_coeff['bert.embeddings.position_embeddings.weight']
normal_model_copy['bert.embeddings.token_type_embeddings.weight'] = small_model_coeff['bert.embeddings.token_type_embeddings.weight']
normal_model_copy['bert.embeddings.token_type_embeddings.weight'] = small_model_coeff['bert.embeddings.token_type_embeddings.weight']
normal_model_copy['bert.embeddings.LayerNorm.weight'] = small_model_coeff['bert.embeddings.LayerNorm.weight']
normal_model_copy['bert.embeddings.LayerNorm.bias'] = small_model_coeff['bert.embeddings.LayerNorm.bias']

copy_weight_indices = ['0, 3,5']
key_ids = list(normal_model_coeff.keys())[6:]
for key in key_ids:
    if '0' in key :
        small_key = key
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
        print(key, small_key)
    if '3' in key :
        small_key = key
        small_key = small_key.replace('3', '1')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    if '5' in key :
        small_key = key
        small_key = small_key.replace('5', '2')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    if 'cls' in key :
        small_key = key
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    else:
        if 'bias' in key:
            normal_model_copy[key] = torch.zeros(normal_model_copy[key].shape)
        elif 'intermediate' in key:
            interm_matrix = normal_model_copy[key]
            U, s, Vt = torch.linalg.svd(interm_matrix, full_matrices=False)
            A = (U[:, :m]@torch.diag(s[:m]))
            normal_model_copy[key] = A
        elif 'output' in key :
            if 'dense' in key and 'attention' in key:
                normal_model_copy[key] =torch.eye(normal_model_copy[key].shape[0]) 
            elif 'LayerNorm' in key :
                normal_model_copy[key] = torch.ones(normal_model_copy[key].shape)
                
            else:
                normal_model_copy[key] = torch.linalg.inv(A.T @ A) @ A.T
            
        #normal_model_copy[key] = torch.ones(normal_model_copy[key].shape) + torch.normal(0, 0.001,normal_model_copy[key].shape )
torch.save(normal_model_copy, 'copylayers_identity_init.bin')


normal_model_file = 'output_baseline_6L_768H/checkpoints/checkpoint-00000000/pytorch_model.bin'
normal_model_copy = torch.load(normal_model_file, map_location=torch.device('cpu'))
normal_model_copy['bert.embeddings.word_embeddings.weight'] = small_model_coeff['bert.embeddings.word_embeddings.weight']
normal_model_copy['bert.embeddings.position_embeddings.weight'] = small_model_coeff['bert.embeddings.position_embeddings.weight']
normal_model_copy['bert.embeddings.token_type_embeddings.weight'] = small_model_coeff['bert.embeddings.token_type_embeddings.weight']
normal_model_copy['bert.embeddings.token_type_embeddings.weight'] = small_model_coeff['bert.embeddings.token_type_embeddings.weight']
normal_model_copy['bert.embeddings.LayerNorm.weight'] = small_model_coeff['bert.embeddings.LayerNorm.weight']
normal_model_copy['bert.embeddings.LayerNorm.bias'] = small_model_coeff['bert.embeddings.LayerNorm.bias']

key_ids = list(normal_model_coeff.keys())[6:]
for key in key_ids:
    if '0' in key :
        small_key = key
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
        print(key, small_key)
    if '3' in key :
        small_key = key
        small_key = small_key.replace('3', '1')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    if '5' in key :
        small_key = key
        small_key = small_key.replace('5', '2')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    if 'cls' in key :
        small_key = key
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
torch.save(normal_model_copy, 'copylayers_default_init.bin')


normal_model_file = 'output_baseline_6L_768H/checkpoints/checkpoint-00000000/pytorch_model.bin'
normal_model_copy = torch.load(normal_model_file, map_location=torch.device('cpu'))

normal_model_copy['bert.embeddings.word_embeddings.weight'] = small_model_coeff['bert.embeddings.word_embeddings.weight']
normal_model_copy['bert.embeddings.position_embeddings.weight'] = small_model_coeff['bert.embeddings.position_embeddings.weight']
normal_model_copy['bert.embeddings.token_type_embeddings.weight'] = small_model_coeff['bert.embeddings.token_type_embeddings.weight']
normal_model_copy['bert.embeddings.token_type_embeddings.weight'] = small_model_coeff['bert.embeddings.token_type_embeddings.weight']
normal_model_copy['bert.embeddings.LayerNorm.weight'] = small_model_coeff['bert.embeddings.LayerNorm.weight']
normal_model_copy['bert.embeddings.LayerNorm.bias'] = small_model_coeff['bert.embeddings.LayerNorm.bias']

key_ids = list(normal_model_coeff.keys())[6:]
for key in key_ids:
    if '0' in key :
        small_key = key
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
        print(key, small_key)
    if '1' in key :
        small_key = key
        #small_key = small_key.replace('1', '1')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    if '2' in key :
        small_key = key
        #small_key = small_key.replace('2', '2')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    if '3' in key :
        small_key = key
        small_key = small_key.replace('3', '0')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
        print(key, small_key)
    if '4' in key :
        small_key = key
        small_key = small_key.replace('4', '1')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    if '5' in key :
        small_key = key
        small_key = small_key.replace('5', '2')
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
    if 'cls' in key :
        small_key = key
        normal_model_copy[key] = small_model_coeff[small_key]
        print(key , normal_model_coeff[key].shape)
torch.save(normal_model_copy, 'copylayers_stacked_init.bin')


