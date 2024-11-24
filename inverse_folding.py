import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data
import torch.optim as optim
# Optimizer
from torch.optim.lr_scheduler import StepLR
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
from inverse import AttU_Net, RNADesignData, one_hot_to_sequence, masked_cross_entropy_loss
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
        # if batch_n%100==0:
        #     print('Sequencing number: ', batch_n)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        #u_no_train = postprocess(pred_contacts,
            #seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
            #seq_ori, 0.01, 0.1, 50, 1, True)
        #map_no_train = (u_no_train > 0.5).float()
        return(pred_contacts)
        
    
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
def main():
    
    
    with open('C:\\Users\\1\\Desktop\\UFold\\data\\encoded_data_varied.pkl', 'rb') as handle:
        xxo = pickle.load(handle)
    encoded_bp=xxo['encoded_bp']
    encoded_seq=xxo['encoded_seq']
    encoded_dotb=xxo['encoded_dotb']
    encoded_sub1=xxo['encoded_sub1']
    encoded_sub2=xxo['encoded_sub2']
    #torch.multiprocessing.set_sharing_strategy('file_system')
    #torch.cuda.set_device(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    if not os.path.exists('C:\\Users\\1\\Desktop\\UFold\\results\\save_ct_file'):
        os.makedirs('C:\\Users\\1\\Desktop\\UFold\\dataresults\\save_ct_file')
    if not os.path.exists('C:\\Users\\1\\Desktop\\UFold\\dataresults\\save_varna_fig'):
        os.makedirs('C:\\Users\\1\\Desktop\\UFold\\dataresults\\save_varna_fig')
    config_file = args.config
    test_file = args.test_files

    config = process_config(config_file)
    seed_torch()
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    trainData = RNADesignData(encoded_bp, encoded_seq, encoded_dotb, encoded_sub1, encoded_sub2)
    model1 = AttU_Net(64,16)
    train_loader = DataLoader(trainData, batch_size=1)
    model1 = model1.to(device)
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 1,
              'drop_last': True}
    MODEL_SAVED = 'models/ufold_train_alldata.pt'
    contact_net = FCNNet(img_ch=17)
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device, weights_only=False))
    contact_net = contact_net.to(device)
    contact_net.eval()
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    num_epochs = 500
    model1.train()
    best_loss = 100000000
    #optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer = optim.Adam(model1.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by a factor of 0.1 every 10 epochs
    for epoch in range(num_epochs):
        epoch_loss = 0
        model1.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for inputs, seq in train_loader:
                # Move data to the same device as the model (GPU if available)
                #inputs = inputs.permute(0, 3, 1, 2)
                inputs = tuple(inp.to(device) for inp in inputs)
                seq = seq.to(device)
                ss_struct = inputs[0]
                # Zero the parameter gradients
                # Forward pass
                #inputs = inputs.permute(0, 3, 1, 2)
                output = model1(inputs)
                seqx = one_hot_to_sequence(output)
                test_data = RNASSDataGenerator_seq('C:\\Users\\1\\Desktop\\UFold\\data\\', seqx[0], "1ddy_A")
                test_set = Dataset_FCN(test_data)
                test_generator = data.DataLoader(test_set, **params)
                contact_net.to(device)
                with torch.no_grad():  # No need to calculate gradients for contact_net
                    matx = model_eval_all_test(device, contact_net, test_generator)[:, :len(seqx[0]), :len(seqx[0])]
                # Compute loss
                optimizer.zero_grad()
                loss_u = criterion_bce_weighted(matx, ss_struct[:,:,:,0])
                loss1 = masked_cross_entropy_loss(output, seq, ss_struct)
                lossx = 0.5*loss1+ 0.5*loss_u

                # print(lossx)
                # Backpropagation
                lossx.backward()
                optimizer.step()

                # Accumulate the loss for averaging
                epoch_loss += lossx.item()

                # Update the progress bar and show the current batch loss
                pbar.set_postfix(loss=lossx.item())
                pbar.update(1)


            #print(f"Batch Loss: {lossx.item()}")
            scheduler.step()
        # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}')

            # Save the best model based on loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model1.state_dict(), "best_attention_unet_model.pt")
                print("Best model saved with loss:", best_loss)
    
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()





