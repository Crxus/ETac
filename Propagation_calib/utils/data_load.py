import torch
import pickle
import numpy as np


def getdata(basepath, num):
    '''
    Process the data, which are saved as pickle files
    data format: dict {"uppers": tensor[n, 3], surface nodes of elastomer in FEM simulation, 
            "cps": tensor[num_contact, 3], active nodes (nodes in direct contact) among n nodes between the elastomer and the indenter}
    
    if U r tying to collect ur own data, try to name the files in order, e.g., 0.pkl, 1.pkl, ...

    Parameters:
    basepath: str, path to the data folder
    num: int, number of files

    return: ori_pts: tensor[B, n, 3], surface nodes before indentation
            upper_deform: tensor[B, n, 3], deformation of the surface nodes during indentation
            cps: [B], number of contact points during indentation
    '''
    center = False
    upper_deform = torch.tensor([])
    cps = []
    ori_pts = None
    for i in range(num):
        name = basepath + '/' + str(i) + '.pkl'
        with open(name, 'rb') as f:
            loaded_data = pickle.load(f)  #  "uppers": surface points, "cps": num of contact points
        mid_ups = torch.tensor([])
        for j, item in enumerate(loaded_data):
            if j==0 and i==0: # save the positions before deform
                ori_pts = item['uppers'].unsqueeze(0).clone()
            ups = item['uppers'].unsqueeze(0) #[1, n, 3]
            cp = len(item['cps'])
            mid_ups = torch.cat((mid_ups, ups), dim=0)
            cps.append(cp)

        # centeralize
        mid_ups[:, :, 0] -= torch.min(mid_ups[:, :, 0])
        mid_ups[:, :, 1] -= torch.min(mid_ups[:, :, 1])
        mid_ups[:, :, 2] -= torch.min(mid_ups[:, :, 2])

        if center == False:
            pts = loaded_data[0]['surfaces']
            pts1 = pts.clone()
            pts1[:, 0] -= torch.min(mid_ups[:, :, 0])
            pts1[:, 1] -= torch.min(mid_ups[:, :, 1])
            pts1[:, 2] -= torch.min(mid_ups[:, :, 2])
        deforms = mid_ups - mid_ups[0].unsqueeze(0) # get the deformation, [B, n, 3]
        upper_deform = torch.cat((upper_deform, deforms), dim=0)


    return ori_pts, upper_deform, cps

 