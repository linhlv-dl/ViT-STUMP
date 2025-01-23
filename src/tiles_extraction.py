import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_top_tiles(wsi_tiles, attention_scores, nb_tiles = 5):
    '''
        This function is to extract nb_tiles with highest and nb_tiles with lowest attention score.
        - wsi_tiles: List of tiles (indices) corresponding to the list of attention scores
        - attention_scores: The attention scores provided by the trained model
        - nb_tiles: The number of tiles that we would like to extract from highest/lowest attention scores.
    '''
    attention_1d = attention_scores.squeeze(0)
    indices_sorted = torch.argsort(attention_1d, descending = True)
    sorted_attention_scores = attention_1d[indices_sorted]
    sorted_attention_scores = sorted_attention_scores.numpy()
    sorted_wsi_tiles = wsi_tiles[indices_sorted]

    # Select the top and bottom feature
    selected_scores = np.concatenate((sorted_attention_scores[:nb_tiles],sorted_attention_scores[-nb_tiles:]))
    selected_tiles = np.concatenate((sorted_wsi_tiles[:nb_tiles], sorted_wsi_tiles[-nb_tiles:]))
    # print(selected_scores.shape, selected_tiles.shape)

    # Select the top only
    # selected_scores = sorted_attention_scores[:,:top]
    # selected_scores = np.swapaxes(selected_scores,1,0)
    # selected_tiles = sorted_wsi_tiles[:top,:]
    #print(selected_scores.shape, selected_tiles.shape)

    return selected_tiles, selected_scores

def display_top(tiles, scores, labels, top = 10, save_file = 'top_tiles'):
    nrows = 10
    ncols =  (2 * top) // nrows 
    fig = plt.figure(figsize = (37,17))
    plt.tight_layout()
    for i in range(2 * top):
        tile = tiles[i]
        score = scores[i]
        label = labels[i]
        ax = fig.add_subplot(nrows, ncols, i + 1).set_title(f'{score[0]:.1e}, {label}', fontdict = {'fontsize':11})
        plt.imshow(Image.fromarray(tile))
        
    plt.savefig('exports/OS_to_export_tiles/top_M_tiles/'+ save_file + '_tiles.png')
    #plt.show()
    plt.close(fig)
    return

def extracted_seletected_tiles(all_tiles, exp_tile_indices, patient, export_to = "npz", save_folder = "./tmp"):
    exp_tile_indices = exp_tile_indices.reshape(-1)
    total_tiles = exp_tile_indices.shape[0]
    
    if export_to == "npz":
        file_name = "{}_{}_top_{}_bottom_tiles.npz".format(patient, total_tiles//2, total_tiles//2)
        save_path = os.path.join(save_folder, file_name)
        extracted_tiles = all_tiles[exp_tile_indices,:]
        np.savez_compressed(save_path, extracted_tiles)
    elif export_to == "png":
        try:
            # create the folder
            pt_dir = os.path.join(save_folder, patient)
            os.makedirs(pt_dir)

        except OSError:
            pass

        # Save the png to the folder
        for i, index in enumerate(exp_tile_indices):
            tile_idx = all_tiles[index,:,:,:]
            img = Image.fromarray(tile_idx.astype('uint8')).convert('RGB')

            # create file name and save to png
            if i <= (len(exp_tile_indices) // 2):
                base_name = '{}_top_{}_{}.png'.format(patient, str(i), str(index))
            else:
                base_name = '{}_bot_{}_{}.png'.format(patient, str(i), str(index))
            t_path = os.path.join(pt_dir, base_name)
            img.save(t_path)
    else:
        raise("Can not export the tiles")

    return