import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data
import pickle
# from FCN import FCNNet
from Network import U_Net as FCNNet
from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset,RNASSDataGenerator_seq
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
#from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections
import torch.multiprocessing as mp
from inverse import AttU_Net, RNADesignData, one_hot_to_sequence
from torch.utils.data import Dataset, DataLoader



#import subprocess
args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis = 1).clamp_max(1))
    seq[contact.sum(axis = 1) == 0] = -1
    return seq

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def get_ct_dict(predict_matrix,batch_num,ct_dict):
    
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    #pdb.set_trace()
    #print(seq_name)
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmpp = np.copy(seq_tmp)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    letter='AUCG'
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    #seq = ((seq_tmp+1).squeeze()[:len(seq_letter)],torch.arange(predict_matrix.shape[-1]).numpy()[:len(seq_letter)]+1)
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    dot_file_dict[batch_num] = [(seq_name.replace('/','_'),seq_letter,dot_list[:len(seq_letter)])]
    #pdb.set_trace()
    ct_file_output(ct_dict[batch_num],seq_letter,seq_name,'results/save_ct_file')
    _,_,noncanonical_pairs = type_pairs(ct_dict[batch_num],seq_letter)
    tertiary_bp = [list(x) for x in set(tuple(x) for x in noncanonical_pairs)]
    str_tertiary = []
    for i,I in enumerate(tertiary_bp):
        if i==0:
            str_tertiary += ('(' + str(I[0]) + ',' + str(I[1]) + '):color=""#FFFF00""')
        else:
            str_tertiary += (';(' + str(I[0]) + ',' + str(I[1]) + '):color=""#FFFF00""')

    tertiary_bp = ''.join(str_tertiary)
    #return ct_dict,dot_file_dict
    return ct_dict,dot_file_dict,tertiary_bp

def ct_file_output(pairs, seq, seq_name, save_result_path):

    #pdb.set_trace()
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]-1] = int(I[1])
        #col5[I[1]] = int(I[0]) + 1
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    np.savetxt(os.path.join(save_result_path, seq_name.replace('/','_'))+'.ct', (temp), delimiter='\t', fmt="%s", header='>seq length: ' + str(len(seq)) + '\t seq name: ' + seq_name.replace('/','_') , comments='')

    return

def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]
    # seq_pairs = [[sequence[i[0]],sequence[i[1]]] for i in pairs]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0]-1],sequence[i[1]-1]] in [["A","U"], ["U","A"]]:
            AU_pair.append(i)
        elif [sequence[i[0]-1],sequence[i[1]-1]] in [["G","C"], ["C","G"]]:
            GC_pair.append(i)
        elif [sequence[i[0]-1],sequence[i[1]-1]] in [["G","U"], ["U","G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
        # print(watson_pairs_t, wobble_pairs_t, other_pairs_t)
    return watson_pairs_t, wobble_pairs_t, other_pairs_t


def model_eval_all_test(device, contact_net,test_generator):
    batch_n = 0
    for seq_embeddings, seq_lens, seq_ori, seq_name in test_generator:
    #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        if batch_n%100==0:
            print('Sequencing number: ', batch_n)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
            #seq_ori, 0.01, 0.1, 50, 1, True)
        map_no_train = (u_no_train > 0.5).float()
        return(map_no_train)
        
    

def main():
    
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    if not os.path.exists('results/save_ct_file'):
        os.makedirs('results/save_ct_file')
    if not os.path.exists('results/save_varna_fig'):
        os.makedirs('results/save_varna_fig')
    config_file = args.config
    test_file = args.test_files

    config = process_config(config_file)
    seed_torch()
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    epoches_first = config.epoches_first
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': True}
    MODEL_SAVED = 'models/ufold_train_alldata.pt'
    contact_net = FCNNet(img_ch=17)
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device))
    
    #GGGGGCUCUGUUGGUUCUCCCGCAACGCUACUCUGUUUACCAGGUCAGGUCCGAAAGGAAGCAGCCAAGGCAGAUGACGCGUGUGCCGGGAUGUAGCUGGCAGGGCCCCC
    seqx="UUUUUUUAUAUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUCUUUUUUUUUUUCCUUUU"
    print(seqx)
    test_data = RNASSDataGenerator_seq('data/', seqx, "1ddy_A")
    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)
    contact_net.to(device)
    matx=model_eval_all_test(device,contact_net,test_generator)
    print(matx.shape)
    
    
    
    #pdb.set_trace()
    print('==========Start Loading Pretrained Model==========')
    
    print('==========Finish Loading Pretrained Model==========')
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    print('==========Done!!! Please check results folder for the predictions!==========')

    #print(ss_struct.shape)
            # Compute loss
            #loss = masked_cross_entropy_loss(output, seq, ss_struct)
            
            # Backward pass and optimize
            #loss.backward()
            #optimizer.step()
            
            # Accumulate loss
            #epoch_loss += loss.item()
            
            # Print loss for the current batch
            #print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        #scheduler.step()
        # Print the average loss for the epoch
        
    
    

    

    

    
    
    
    
   

    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()





