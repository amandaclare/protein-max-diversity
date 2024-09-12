import numpy as np
import h5py
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.spatial.distance as dist # dist.cityblock(a,b)
import requests
import gzip
import os
from sklearn.decomposition import PCA
import re

# In Swissprot we have the following:
# number of entries: 568354
# embedding size: 1024

max_tabu_length = 100
worse_thresh = 0.95
max_steps = 200

#------------------------------------------------------------------
# Reading in data

def read_ecs(filename):
    ec_d = {}
    with gzip.open(filename, "rb") as f:
        header = next(f)
        for line in f:
            ws = line.split(b'\t')
            if ws[7] != b'\n':
                ec_d[ws[0].decode()] = [w.decode().strip() for w in ws[7].strip().split(b';')]
    return ec_d


def read_embeddings_proportion(pdict, proportion, filename):
    with h5py.File(filename, "r") as file:
        #print(f"number of entries: {len(file.items())}")
        for sequence_id, embedding in file.items():
            i = random.random()
            if i < proportion:
                pdict[sequence_id] = np.array(embedding)


def read_embeddings_ec(pdict, ec_dict, ec_num, filename):
    with h5py.File(filename, "r") as file:
        #print(f"number of entries: {len(file.items())}")
        for sequence_id, embedding in file.items():
            if sequence_id in ec_dict:
                for num in ec_dict[sequence_id]: # multiple ec nums
                    if num == ec_num:
                        pdict[sequence_id] = np.array(embedding)
                        # got one, so ignore others for this protein
                        break


def read_embeddings_chosen(pdict, ids_filename, filename):
    ids = set()
    with open(ids_filename) as f:
        for line in f:
            ids.add(line.strip())
            
    with h5py.File(filename, "r") as file:
        for sequence_id, embedding in file.items():
            if sequence_id in ids:
                pdict[sequence_id] = np.array(embedding)


#------------------------------------------------------------------
# Tabu search
                
#Finds which element of the set has the minimum distance to all others, and
# returns this element (k,v) and its min distance.
def find_min_dist_elem(chosen):
    vals = [v for (k,v) in chosen]
    dists = dist.cdist(vals, vals, 'cityblock')
    # warning, finding min dists is hard because (x,x) dist is 0 all down diag
    np.fill_diagonal(dists, 1000)
    m = np.min(dists, axis=0)
    min_elem = np.argmin(m)
    k = chosen[min_elem]
    return (k, m[min_elem])


    
def find_replacement(worst_elem, min_dist, result_list, tabu_list, pdict):
    # Find an element from pdict whose min pairwise distance to all
    # items in result_list is more than min_dist, without using elements
    # in tabu list
    #order = list(pdict.keys())
    #random.shuffle(order)
    
    alternatives = []
    
    (best_k, best_dist) = (None, min_dist)
    result_keys, result_vals = zip(*result_list)
    for k in pdict.keys():
        if k not in tabu_list and k not in result_keys:
            d = np.min(dist.cdist([pdict[k]],result_vals, 'cityblock'))

            # If it's good enough, retain it, or if its worse but not
            # awful, put in alternatives list, or ignore
            if d > best_dist:
                #return (k, pdict[k])
                (best_k, best_dist) = (k, d)
            elif d < min_dist and d/min_dist > worse_thresh: # worse but not bad
                alternatives.append(k)
                
    if best_k is not None:
        return (best_k, pdict[best_k])
    elif alternatives != []:
        alt_k = random.choice(alternatives)
        return (alt_k, pdict[k])
    else:
        # no element better and no good enough worse alternative
        return (None, None)
    


# Tabu search. Returns a list of (id, embedding) pairs.
def tabu(pdict, m):
    min_dist_so_far = 0
    tabu_list = []    # list of ids
    result_list = random.sample(pdict.items(), m) # list of pairs
    making_progress = True 

    steps = 0
    
    while making_progress and steps < max_steps:
        print("steps:", steps)

        # find worst element and remove it
        (worst_elem, min_dist) = find_min_dist_elem(result_list)
        if min_dist > min_dist_so_far:
            min_dist_so_far = min_dist
        result_list.remove(worst_elem)

        # Find a replacement (k, embs) for this element if possible
        new_elem = find_replacement(worst_elem, min_dist, result_list, tabu_list, pdict)
        if new_elem[0] is None:
            making_progress = False
        else:
            result_list.append(new_elem)
            # update the tabu list
            tabu_list.append(worst_elem[0])
            while len(tabu_list) > max_tabu_length:
                tabu_list.pop(0)

        steps += 1

    if not making_progress:
        result_list.append(worst_elem)
        
    print(min_dist_so_far)
    return result_list


#-------------------------------------------------------------
# Output and PCA

# Look up the m proteins in UniProt and write out their info, tab-sep
def write_uniprot_details(results_dir, result, m, proportion, iteration):
    with open(f"{results_dir}/diverse_proteins_{m}_{proportion}_{iteration}.csv","w") as f:
        for (k, v) in result:
            # get basic info (taxon, length, name)
            r = requests.get(f"https://rest.uniprot.org/uniprotkb/{k}.tsv")
            if r.status_code == 200:
                lines = r.text.split("\n")
                fields = lines[1].split("\t")
                f.write('\t'.join(fields))
            else:
                sys.stderr.write(f'Lookup of tsv file for {k} at uniprot failed')
            # get sequence
            r = requests.get(f"https://rest.uniprot.org/uniprotkb/{k}.fasta")
            if r.status_code == 200:
                lines = r.text.split("\n")
                header = lines[0]
                seq = ''.join([line.strip() for line in lines[1:]])
                f.write('\t'+seq)
            else:
                sys.stderr.write(f'Lookup of sequence for {k} at uniprot failed'
)
            # write out the vector too just in case
            print('\t'+str(list(v)), file=f)




def PCA_overlay_analysis(results_dir, pdict, result, m, proportion, iteration):
    data = [v for (k,v) in result]
    names = [k for (k,v) in result]
    all_data = list(pdict.values()) 
    pca = PCA(n_components=2)
    transformed_all_data = pca.fit_transform( all_data )
    plt.scatter(transformed_all_data[:,0], transformed_all_data[:,1])
    transformed_data = pca.transform(data)
    plt.scatter(transformed_data[:,0], transformed_data[:,1], color="r")
    for (i,txt) in enumerate(names):
        plt.annotate(txt,(transformed_data[i,0],transformed_data[i,1]))
        
    plt.savefig(f"{results_dir}/diverse_proteins_{m}_{proportion}_{iteration}.png",dpi=400)
    plt.close()
    
    # see the first two component weightings
    with open(f"{results_dir}/diverse_proteins_{m}_{proportion}_{iteration}_weights.csv","w") as f:
        print(list(pca.components_[0,:]), file=f)
        print(list(pca.components_[1,:]), file=f)

    print("Explained variance ratio",pca.explained_variance_ratio_)

    
#-----------------------------------------------------------------------    
# Main

def main():
    # protein embeddings dictionary
    pdict = {}

    # Find m most diverse items
    m = 20

    # input data
    filename = "per-protein.h5"

    # repeat a number of times because of stochasticity effects
    repeat = 10
    
    # Read in a proportion or all of the data
    # What proportion of the data to use (e.g. 0.01 = 1%, 1 = 100%)
    #proportion = 1 
    #read_embeddings_proportion(pdict, proportion, filename)
    #results_dir = "results_m"+str(m)
    
    # or 
    # Read in only the data in a given EC category
    #proportion = 1 
    #ec_num = "2.7.7.7" #"2.7.7.7"  
    #results_dir = "results_m" + str(m) + "_E"+ ec_num
    #ec_filename = "swissprot-compressed_true_download_true_fields_accession_2Creviewed_2C-2023.01.19-15.42.19.73.tsv.gz"
    #ec_dict = read_ecs(ec_filename)
    #read_embeddings_ec(pdict, ec_dict, ec_num, filename)

    # or
    # Read in only the proteins with chosen ids
    proportion = 1 
    read_embeddings_chosen(pdict, "toxins.txt", filename)
    results_dir = "results_m"+str(m)+"_toxins"
    
    # Make the results dir if needed
    os.makedirs(results_dir, exist_ok=True)    
    
    print(len(pdict),"items")
        
    # Try the search a number of times because of random effects 
    for i in range(repeat):
        result = tabu(pdict, m)
        print([k for (k,v) in result])
        # Report what UniProt has to say about them
        write_uniprot_details(results_dir, result, m, proportion, i)
        PCA_overlay_analysis(results_dir, pdict, result, m, proportion, i)

    # Lets compare with randomly generated sets
    for i in range(repeat):
        random_result = [(k,pdict[k]) for k in random.sample(list(pdict.keys()), m)]
        PCA_overlay_analysis(results_dir, pdict, random_result, m, proportion, 100+i)

        
if __name__ == "__main__":
    main()
    
