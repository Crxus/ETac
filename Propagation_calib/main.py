import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from utils.dataset import allData
from utils.data_load import getdata
from utils.data_process import estimate_normals_pca, process
from utils.models import LinearModel, ResdualNet
import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def train_models(linearModel, resdualNet, dataset, optimizer, epochs=[400, 1500], batch_size=64, train_ratio=0.7, save_path='checkpoints/0402/', device = 'cuda'):
    min_loss = 1e2
    res_min_loss = 1e2
    saved_epoch = 0
    train_len = int(len(dataset) * train_ratio)
    max_grad_norm = 1.0
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    torch_state = torch.get_rng_state()
    print("PyTorch RNG State:", torch_state)
    torch.save(torch_state, save_path + "torch_rng_state.pth")

    # phase1, train linear model only
    for epoch in range(epochs[0]):
        linearModel.train()
        total_loss = 0
        for active_mask, init_pos, active_deform, data, deformation, to_active in train_loader:  
            active_mask = active_mask.to(device)
            init_pos = init_pos.float().to(device)
            active_deform = active_deform.to(torch.float32).to(device)
            data = data.float().to(device)   # [B, n, 9]
            deformation = deformation.float().to(device)         # [B, n, 3]
            to_active = to_active.float().to(device)         # [B, n, 3]

            b, _, _ = data.shape
            optimizer.zero_grad()

            mask1 = active_mask.unsqueeze(-1).expand(-1, -1, 3)  #[B, n, 3]
            pred = linearModel(init_pos, init_pos, active_deform)  # [B, n, 3]
            pred = torch.where(mask1, torch.zeros_like(pred), pred) #set the position of active nodes as 0 displacement, since we just calculate propagation to passive nodes

            ground = torch.where(mask1, torch.zeros_like(deformation), deformation)

            mse = loss_fn(pred, ground)
            loss = torch.sqrt(mse) * 1000
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        linearModel.eval()
        total_val_loss = 0
        with torch.no_grad():
            for active_mask, init_pos, active_deform, data, deformation, to_active in val_loader:  
                active_mask = active_mask.to(device)
                init_pos = init_pos.float().to(device)
                active_deform = active_deform.to(torch.float32).to(device)
                data = data.float().to(device)   # [B, n, 9]
                deformation = deformation.float().to(device)         # [B, n, 3]
                to_active = to_active.float().to(device)

                b, _, _ = data.shape
                optimizer.zero_grad()

                mask1 = active_mask.unsqueeze(-1).expand(-1, -1, 3)  #[B, n, 3]
                pred = linearModel(init_pos, init_pos, active_deform)  # [B, n, 3]
                pred = torch.where(mask1, torch.zeros_like(pred), pred) #set the position of active nodes as 0 displacement, since we just calculate propagation to passive nodes

                ground = torch.where(mask1, torch.zeros_like(deformation), deformation)

                mse = loss_fn(pred, ground)
                loss = torch.sqrt(mse) * 1000

                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"[Epoch {epoch+1}/{epochs[0]}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # save the best model
        if avg_val_loss < min_loss:
            #if min_loss - avg_val_loss > 1e-3*min_loss:
            if min_loss - avg_val_loss >= 1e-2 * min_loss or min_loss>1:
                min_loss = avg_val_loss
                torch.save(linearModel.state_dict(), save_path + "LinearModel.pth")
                saved_epoch = epoch
                print(f"Phase1: Saved new best model at epoch {epoch+1} with Val Loss: {avg_val_loss:.6f}")
        if epoch - saved_epoch>=50:
            break
    
    linearModel.load_state_dict(torch.load(save_path + "LinearModel.pth"))


    #phase 2: uni training
    #pn.load_state_dict(torch.load(save_paths[1]))
    resdualNet.train()
    linearModel.train()
    for param in linearModel.parameters():
        param.requires_grad = True
    for epoch in range(epochs[1]):
        total_loss = 0
        for active_mask, init_pos, active_deform, data, deformation, to_active in train_loader:  
            active_mask = active_mask.to(device)
            init_pos = init_pos.float().to(device)
            active_deform = active_deform.to(torch.float32).to(device)
            data = data.float().to(device)   # [B, n, 9]
            deformation = deformation.float().to(device)         # [B, n, 3]
            to_active = to_active.float().to(device)

            b, _, _ = data.shape
            optimizer.zero_grad()

            mask1 = active_mask.unsqueeze(-1).expand(-1, -1, 3)  #[B, n, 3]
            pred = linearModel(init_pos, init_pos, active_deform)  # [B, n, 3]
            pred = torch.where(mask1, torch.zeros_like(pred), pred) #set the position of active nodes as 0 displacement, since we just calculate propagation to passive nodes
            residue = resdualNet(data) #[B, n, 3]
            residue = torch.where(mask1, torch.zeros_like(residue), residue)
            res = pred + residue
            ground = torch.where(mask1, torch.zeros_like(deformation), deformation) # same reason as above

            res1 = res.clone() * to_active.unsqueeze(-1)
            ground1 = ground.clone() * to_active.unsqueeze(-1)

            loss1 = torch.sqrt(loss_fn(res1, ground1)) * 1000

            loss = 5*loss1 #+ 0.5*loss0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(resdualNet.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        linearModel.eval()
        total_val_loss = 0

        with torch.no_grad():
            for active_mask, init_pos, active_deform, data, deformation, to_active in val_loader:  
                active_mask = active_mask.to(device)
                init_pos = init_pos.float().to(device)
                active_deform = active_deform.to(torch.float32).to(device)
                data = data.float().to(device)   # [B, n, 9]
                deformation = deformation.float().to(device)         # [B, n, 3]
                to_active = to_active.float().to(device)

                b, _, _ = data.shape
                optimizer.zero_grad()

                mask1 = active_mask.unsqueeze(-1).expand(-1, -1, 3)  #[B, n, 3]
                pred = linearModel(init_pos, init_pos, active_deform)  # [B, n, 3]
                pred = torch.where(mask1, torch.zeros_like(pred), pred) #set the position of active nodes as 0 displacement, since we just calculate propagation to passive nodes
                residue = resdualNet(data) #[B, n, 3]
                residue = torch.where(mask1, torch.zeros_like(residue), residue)
                res = pred + residue
                ground = torch.where(mask1, torch.zeros_like(deformation), deformation) # same reason as above

                '''
                The goal of the training process is to minimize the error of the predicted displacements of passive nodes, 
                especially those are closer to the active nodes, since they are more likely to be influenced by the active nodes.
                '''
                res_ = res.clone() * to_active.unsqueeze(-1)
                ground_ = ground.clone() * to_active.unsqueeze(-1)
                mse = loss_fn(res_, ground_)

                loss = torch.sqrt(mse) * 1000

                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"[Epoch {epoch+1}/{epochs[1]}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # save the best model
        if avg_val_loss < res_min_loss:
            #if min_loss - avg_val_loss > 1e-3*min_loss:
            if res_min_loss - avg_val_loss >= 1e-2 * res_min_loss or res_min_loss>1:
                res_min_loss = avg_val_loss
                torch.save(linearModel.state_dict(), save_path + "LinearModel.pth")
                torch.save(resdualNet.state_dict(), save_path + "ResidualNet.pth")
                saved_epoch = epoch
                print(f"Phase2: Saved new best model at epoch {epoch+1} with Val Loss: {avg_val_loss:.6f}")

def test(model, net, val_loader, device='cuda'):
    '''
    Return:
    avg_val_loss: float, average validation loss
    pre_displacements: tensor, predicted displacements of all nodes, [B, n, 3]
    truth: tensor, ground truth displacements of all nodes, [B, n, 3]
    original_pts: tensor, undeformed positions of all nodes, [B, n, 3]
    active_masks: tensor, bool, which nodes are active nodes, [B, n], for visulization
    '''
    net.eval()
    model.eval()
    total_val_loss = 0
    loss_fn = nn.MSELoss()
    pre_displacements = None
    truth = None
    original_pts = None
    active_masks = None
    #loss_fn = nn.L1Loss()
    with torch.no_grad():
        for active_mask, init_pos, active_deform, data, deformation, to_active in val_loader:  
            active_mask = active_mask.to(device)
            init_pos = init_pos.float().to(device)
            active_deform = active_deform.to(torch.float32).to(device)
            data = data.float().to(device)   # [B, n, 9]
            deformation = deformation.float().to(device)         # [B, n, 3]
            to_active = to_active.float().to(device) 

            b, _, _ = data.shape

            mask1 = active_mask.unsqueeze(-1).expand(-1, -1, 3)  #[B, n, 3]
            pred = model(init_pos, init_pos, active_deform)  # [B, n, 3]
            pred = torch.where(mask1, torch.zeros_like(pred), pred) #set the position of active nodes as 0 displacement, since we just calculate propagation to passive nodes
            residue = net(data) #[B, n, 3]
            residue = torch.where(mask1, torch.zeros_like(residue), residue)
            res_ = pred + residue

            ground = torch.where(mask1, torch.zeros_like(deformation), deformation) # same reason as above

            est = res_
            if pre_displacements==None:
                pre_displacements = est
                truth = deformation
                original_pts = init_pos
                active_masks = active_mask
            else:
                pre_displacements = torch.cat((pre_displacements, est))
                truth = torch.cat((truth, deformation))
                original_pts = torch.cat((original_pts, init_pos))
                active_masks = torch.cat((active_masks, active_mask))


            mse = loss_fn(est, ground)
            loss = torch.sqrt(mse) * 1000
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    return avg_val_loss, pre_displacements, truth, original_pts, active_masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the deformation propagation model.')

    parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False, help='Whether to run a trained model or train a new model')
    parser.add_argument('--data_id', type=str, default='1', help='Path id to the training data folder')
    parser.add_argument('--num_files', type=int, default=200, help='Number of data files to load, e.g., 200 for 0.pkl to 199.pkl')
    parser.add_argument('--curved', type=str2bool, nargs='?', const=True, default=True, help='Whether the elastomer is curved')
    parser.add_argument('--save_id', type=str, default='0415', help='Path id for saving the checkpoints and visualization data')

    args = parser.parse_args()
    data_id = args.data_id
    basepath = 'data/calib_data/' + data_id
    ckpt_path = 'checkpoints/' + args.save_id + '/'
    num_files = args.num_files
    test_ = args.test
    curved = args.curved

    ori_pts, upper_deform, cps = getdata(basepath, num=num_files) # ori_pts: [B, n, 3], upper_deform: [B, n, 3]
    pts, deformation, original_pts = process(upper_deform, ori_pts, curved=curved) # pts: [B, m, 3], deformation: [B, m, 3], original_pts: [1, m, 3]
    meanx = torch.mean(pts[0, :, 0])
    meany = torch.mean(pts[0, :, 1])
    meanz = torch.mean(pts[0, :, 2])
    center_pt = torch.stack([meanx, meany, meanz])

    mask = torch.max(torch.abs(deformation[:, :, 2]), dim=1).values * 1e3 <= 4.0
    deformation = deformation[mask]  # [B, m, 3]
    origin_pts = original_pts.clone().repeat(deformation.shape[0], 1, 1)  # [B, m, 3]

    # get the normals of surface nodes
    if curved:
        normals = estimate_normals_pca(origin_pts[0], inner=True)  # [m, 3]
        cps_ = (torch.tensor(cps)[mask]).numpy().tolist()  
        center_direct = center_pt.unsqueeze(0) - origin_pts[0]  # [m, 3]
        flip_mask = torch.sum(normals * center_direct, dim=-1) < 0
        normals[flip_mask] = -normals[flip_mask]
    else:
        normals = torch.zeros_like(origin_pts[0])
        normals[:, 2] = -1.0  # for flat elastomer, normals are all facing downwards
    dataset = allData(deformation, origin_pts[0], normals)
    dataset.prerocess(cps_)

    linear_model = LinearModel().cuda()
    resdual_net = ResdualNet().cuda()

    if test_:
        linear_ckpt = ckpt_path + 'LinearModel.pth'
        resdual_ckpt = ckpt_path + 'ResidualNet.pth'
        linear_model.load_state_dict(torch.load(linear_ckpt))
        resdual_net.load_state_dict(torch.load(resdual_ckpt))
        linear_model.eval()
        resdual_net.eval() 
        var_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        avg_val_loss, res, truth, ori, active_masks = test(linear_model, resdual_net, var_loader)
        print(f"Test Loss: {avg_val_loss:.6f}")

        vis_data_dir = os.path.join('data', 'vis_data', args.save_id)
        os.makedirs(vis_data_dir, exist_ok=True)
        vis_data_path = os.path.join(vis_data_dir, 'test_results.pth')
        torch.save({'res': res.cpu(), 'truth': truth.cpu(), 'ori': ori.cpu(), 'active_masks': active_masks.cpu()}, vis_data_path)

    
    else:
        optimizer = torch.optim.Adam(list(linear_model.parameters()) + list(resdual_net.parameters()), lr=1e-3, weight_decay=1e-5)
        train_models(linear_model, resdual_net, dataset, optimizer, epochs=[400, 1500], batch_size=64, train_ratio=0.7, save_path=ckpt_path, device = 'cuda')
    # optimizer = torch.optim.Adam(list(linear_model.parameters()) + list(resdual_net.parameters()), lr=1e-3, weight_decay=1e-5)
    # train_models(linear_model, resdual_net, dataset, optimizer)

    pass