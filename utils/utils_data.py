"""
Code based on:
Maziarka et al "Molecule Attention Transformer" -> https://github.com/ardigen/MAT
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset

from tqdm import tqdm

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
IntTensor = torch.IntTensor
DoubleTensor = torch.DoubleTensor



def load_data_from_df(dataset_path, add_dummy_node=True, one_hot_formal_charge=True, use_data_saving=True,
                      smiles_column='canonical_smiles'):
    """Load and featurize data stored in a CSV file.

    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
        use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                is present, the features will be saved after calculations. Defaults to True.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    feat_stamp = f'{"_dn" if add_dummy_node else ""}{"_ohfc" if one_hot_formal_charge else ""}'
    extension = '.'+dataset_path.split('.')[-1]
    feature_path = dataset_path.replace(extension, f'{feat_stamp}.p')
    if use_data_saving and os.path.exists(feature_path):
        logging.info(f"Loading features stored at '{feature_path}'")
        x_all = pickle.load(open(feature_path, "rb"))
        return x_all

    data_df = pd.read_csv(dataset_path)

    data_x = data_df[smiles_column].values


    unique_smiles = list(set(data_x))
    x_unique = load_data_from_smiles(unique_smiles, add_dummy_node=add_dummy_node,
                                         one_hot_formal_charge=one_hot_formal_charge)
    comp_feat_dict={}
    for smiles,feat in zip(unique_smiles,x_unique):
        comp_feat_dict[smiles]=feat
    
    x_all = [comp_feat_dict[smi] for smi in tqdm(data_x)]
    
    if use_data_saving and not os.path.exists(feature_path):
        logging.info(f"Saving features at '{feature_path}'")
        pickle.dump((x_all), open(feature_path, "wb"))

    return x_all

def load_data_from_smiles(x_smiles:list, add_dummy_node=True, one_hot_formal_charge=False):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all = []

    logging.info('Load and featurize data from lists of SMILES strings.')
    for smiles in tqdm(x_smiles):   
        try:
            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=100)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            x_all.append([afm, adj, dist])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    return x_all


def featurize_mol(mol, add_dummy_node, one_hot_formal_charge):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """
    def __init__(self, x_smiles, x_geneExpression, y_target, 
                 dose, time, index):
        self.node_features = x_smiles[0]
        self.adjacency_matrix = x_smiles[1]
        self.distance_matrix = x_smiles[2]
        self.x_geneExpression = x_geneExpression
        self.y_target = y_target
        self.dose = dose
        self.time = time
        self.index = index

class Molecule_wo_dosetime:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x_smiles, x_geneExpression, y_target, index):
        self.node_features = x_smiles[0]
        self.adjacency_matrix = x_smiles[1]
        self.distance_matrix = x_smiles[2]
        self.x_geneExpression = x_geneExpression
        self.y_target = y_target
        self.index = index

class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features:
       adjacency matrices, node features, distance matrices, and labels.
    """
    
    adjacency_list, distance_list, features_list = [], [], []
    SMILES=[]
    x_geneExpression = []
    y_target = []
    dose = []
    time = []

    max_size = 0
    for molecule in batch:
        x_geneExpression.append(molecule.x_geneExpression)
        y_target.append(molecule.y_target)
        dose.append(molecule.dose)    
        time.append(molecule.time)
        
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]
            
    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        distance_list.append(pad_array(molecule.distance_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))

    return [FloatTensor(features) for features in (np.array(adjacency_list), np.array(features_list), 
                                                   np.array(distance_list), np.array(x_geneExpression), 
                                                   np.array(y_target), np.array(dose), np.array(time))]


def construct_dataset(x_smiles, x_geneExpression, y_target, doses, times):
    """Construct a MolDataset object from the provided data.

    Args:
        x_smiles (list): A list of molecule features.
        x_geneExpression (list): A list of input basal gene expressions
        y_target (list): A list of the corresponding targets.
        doses (list): A list of the corresponding doses.
        times (list): A list of the corresponding times.

    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[1], data[2],data[3], data[4], i)
              for i, data in enumerate(zip(x_smiles, x_geneExpression, y_target, doses, times))]
    return MolDataset(output)


def construct_loader(x_smiles, x_geneExpression, y_target, doses, times, 
                     batch_size, valid_ratio=0, test_ratio=0, shuffle=True,seed=42):
    """Construct a data loader for the provided data.

    Args:
        x_smiles (list): A list of molecule features.
        x_geneExpression (list): A list of input basal gene expressions
        y_target (list): A list of the corresponding targets.
        doses (list): A list of the corresponding doses.
        times (list): A list of the corresponding times.
        batch_size (int): The batch size.
        shuffle (bool): If True the data will be loaded in a random order. Defaults to True.

    Returns:
        A DataLoader object that yields batches of condition & padded molecule features.
    """
    data_set = construct_dataset(x_smiles, x_geneExpression, y_target, doses, times)
    num_val=int(len(data_set)*valid_ratio)
    num_test=int(len(data_set)*test_ratio)
    num_train=len(data_set)-num_val-num_test
    train_set, valid_set, test_set= torch.utils.data.random_split(data_set,
                                            [num_train,num_val,num_test],
                                            generator=torch.Generator().manual_seed(seed))
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                         batch_size=batch_size,
                                         collate_fn=mol_collate_func,
                                         shuffle=shuffle)
    if valid_ratio:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                         batch_size=batch_size,
                                         collate_fn=mol_collate_func,
                                         shuffle=False)
    else: valid_loader=None
    if test_ratio:
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                         batch_size=batch_size,
                                         collate_fn=mol_collate_func,
                                         shuffle=False)
    else: test_loader=None
    return (train_loader, valid_loader, test_loader)
