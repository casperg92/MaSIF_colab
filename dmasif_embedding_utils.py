import os
import torch

from Bio.PDB import *
from torch_geometric.loader import DataLoader
import argparse
import shutil
from helper import *
import shutil
from typing import Tuple

# Custom data loader and model
from data import load_protein_pair
from model import dMaSIF
from data_iteration import iterate

def create_folder(folder_path: str):
    """
    Create a new folder at the specified path, removing any existing folder and its content at that location.
    
    Args:
        folder_path (str): The path to the new folder.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def get_model(model_resolution: float = 0.7, patch_radius: int = 12) -> Tuple[str, int]:
    """
    Get the path and supsampling value of the pre-trained dMaSIF model based on patch radius and model resolution. 
    There are 4 pre-trained models made available by the authors, depending on the model
    
    Args:
        root_dir (str): The base directory where the models folder is located.
        model_resolution (float, optional): The resolution of the model. Defaults to 0.7.
        patch_radius (int, optional): The patch radius of the model. Defaults to 12.
        
    Returns:
        Tuple[str, int]: A tuple containing the path of the model and the supsampling value.

    Raises:
        ValueError: If the patch radius or model resolution provided do not match any pre-trained model configuration.
    """

    model_config = {(9, 1): ('dMaSIF_site_3layer_16dims_9A_100sup_epoch64', 100),
                    (9, 0.7): ('dMaSIF_site_3layer_16dims_9A_0.7res_150sup_epoch85', 150),
                    (12, 1): ('dMaSIF_site_3layer_16dims_12A_100sup_epoch71', 100),
                    (12, 0.7): ('dMaSIF_site_3layer_16dims_12A_0.7res_150sup_epoch59', 100)}
    model_path, supsampling = model_config.get((patch_radius, model_resolution), (None, None))
    if model_path is None:
        raise ValueError("Invalid patch radius or model resolution")
    model_path =  f"models/{model_path}"   
    return model_path, supsampling

def generate_descr(model_path:str, output_path:str, pdb_file:str, npy_directory:str, radius:float, resolution:float,supsampling:int):
    """
    Generate descriptors for a dMaSIF site model
    
    Args:
        model_path (str): The path to the pre-trained dMaSIF site model.
        output_path (str): The directory where the output descriptor files will be stored.
        pdb_file (str): The path to the pdb file for which the descriptor is to be generated.
        npy_directory (str): The directory where the npy file is located.
        radius (float): The radius of the model.
        resolution (float): The resolution of the model.
        supsampling (int): The value of supsampling used for the model.
    """
    parser = argparse.ArgumentParser(description="Network parameters")
    parser.add_argument("--experiment_name", type=str, default=model_path)
    parser.add_argument("--use_mesh", type=bool, default=False)
    parser.add_argument("--embedding_layer",type=str,default="dMaSIF")
    parser.add_argument("--curvature_scales",type=list,default=[1.0, 2.0, 3.0, 5.0, 10.0])
    parser.add_argument("--resolution",type=float,default=resolution)
    parser.add_argument("--distance",type=float,default=1.05)
    parser.add_argument("--variance",type=float,default=0.1)
    parser.add_argument("--sup_sampling", type=int, default=supsampling)
    parser.add_argument("--atom_dims",type=int,default=6)
    parser.add_argument("--emb_dims",type=int,default=16)
    parser.add_argument("--in_channels",type=int,default=16)
    parser.add_argument("--orientation_units",type=int,default=16)
    parser.add_argument("--unet_hidden_channels",type=int,default=8)
    parser.add_argument("--post_units",type=int,default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--radius", type=float, default=radius)
    parser.add_argument("--k",type=int,default=40)
    parser.add_argument("--dropout",type=float,default=0.0)
    parser.add_argument("--site", type=bool, default=True) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--search",type=bool,default=False) 
    parser.add_argument("--single_pdb",type=str,default=pdb_file)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_rotation",type=bool,default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--single_protein",type=bool,default=True) 
    parser.add_argument("--no_chem", type=bool, default=False)
    parser.add_argument("--no_geom", type=bool, default=False)
    args = parser.parse_args("")
 
    # Ensure reproducibility:
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    # Load test dataset:
    test_dataset = [load_protein_pair(args.single_pdb,Path(npy_directory), single_pdb=True)]
    test_pdb_ids = [args.single_pdb]

    # PyTorch geometric expects an explicit list of "batched variables":
    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=batch_vars)

    net = dMaSIF(args)
    net.load_state_dict(torch.load(model_path,map_location=args.device)["model_state_dict"])
    net = net.to(args.device)

    # Perform one pass through the data:
    info = iterate(
        net,
        test_loader,
        None,
        args,
        test=True,
        save_path=Path(output_path),
        pdb_ids=test_pdb_ids,
    )
    return info


def protonate_pdb(reduce_dir:str, target_pdb:str) -> str: 
    """
    Protonates a pdb file using the reduce software.

    Args:
        reduce_dir (str): The directory to save the protonated protein PDB file.
        target_pdb (str): The path to the target pdb file to be protonated.

    Returns:
        str: The path to the protonated pdb file.
    """    
    target_name = os.path.basename(target_pdb)
    tmp_pdb =  os.path.join(reduce_dir, 'tmp.pdb')
    tmp1_pdb =  os.path.join(reduce_dir, 'tmp1.pdb')
    tmp2_pdb =  os.path.join(reduce_dir, 'tmp2.pdb')

    shutil.copy(target_pdb, tmp_pdb)

    # Remove protons if there are any
    try:
        os.system(f'reduce -Trim -Quiet {tmp_pdb} > {tmp1_pdb}')
    except:
        print(f"Failed to remove protons from {target_name}")

    # Add protons
    try:
        os.system(f'reduce -HIS -Quiet {tmp1_pdb} > {tmp2_pdb}')
    except:
        print(f"Failed to add protons to {target_name}")

    reduced_pdb = os.path.join(reduce_dir, f'{target_name}')
    shutil.copyfile(tmp2_pdb, reduced_pdb)   
    return reduced_pdb
