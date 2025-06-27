import os
import copy
import torch
import logging
import warnings
import numpy as np
import statistics
from datetime import datetime
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.optim import Adam
from parser_args import get_args
from torchvision import transforms
from torch.utils.data import BatchSampler, RandomSampler, DataLoader

from preprocess.data import StandardScaler
from preprocess.featurization_2D import BatchMolGraph, MolGraph, get_atom_fdim, get_bond_fdim
from preprocess.utils import get_class_sizes
from preprocess.data_3d import DataProcessor
from preprocess.bert_date import MolecularDataProcessor
from preprocess.data_image import (load_or_generate_images, generate_labels_file, ImageDataset,
                                   load_filenames_and_labels, process_dataset)

from utils.dataset import get_data, split_data, MoleculeDataset, InMemoryDataset
from utils.evaluate import eval_rocauc, eval_rmse

from models.multi_model import Multi_modal

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4
warnings.filterwarnings('ignore')

IMAGEMOL_CONFIG = {
    'size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


def prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, image_data, device):
    # Sequence data processing
    input_ids = seq_data[idx].to(device)
    attention_mask = seq_mask[idx].to(device)

    # 2D molecular map data processing
    mol_batch = MoleculeDataset([gnn_data[i] for i in idx])
    smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()

    mol_graphs = []
    for smiles in smiles_batch:
        try:
            mol_graph = MolGraph(smiles, args)
            mol_graphs.append(mol_graph)
        except Exception as e:
            print(f"Error processing molecule {smiles}: {str(e)}")
            continue
    gnn_batch = BatchMolGraph(mol_graphs, args)

    # 3D geometry data processing
    geo_gen = geo_data.get_batch(idx)
    edge_batch1, edge_batch2 = [], []
    node_id_all = [geo_gen[0].batch, geo_gen[1].batch]
    for i in range(geo_gen[0].num_graphs):
        edge_batch1.append(torch.ones(geo_gen[0][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
        edge_batch2.append(torch.ones(geo_gen[1][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
    edge_id_all = [torch.cat(edge_batch1), torch.cat(edge_batch2)]

    # Image data processing
    if isinstance(image_data[0][0], torch.Tensor):
        image_batch = torch.stack([image_data[i][0] for i in idx]).to(device)
    else:
        image_transform = transforms.Compose([
            transforms.Resize((IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGEMOL_CONFIG['mean'],
                std=IMAGEMOL_CONFIG['std']
            )
        ])
        image_batch = torch.stack([image_transform(image_data[i][0]) for i in idx]).to(device)

    # Target value processing
    mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).to(device)
    targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'gnn_batch': gnn_batch,
        'features_batch': features_batch,
        'geo_gen': geo_gen,
        'node_id_all': node_id_all,
        'edge_id_all': edge_id_all,
        'image_batch': image_batch,
        'mask': mask,
        'targets': targets,
        'smiles': smiles_batch
    }


def train(args, model, optimizer, train_idx_loader, seq_data, seq_mask, datas, data_3d, image_data, device):

    model.train()

    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0
    batch_count = 0

    all_batch_preds = []
    all_batch_labels = []

    for idx in tqdm(train_idx_loader):
        model.zero_grad()
        optimizer.zero_grad()
        batch_data = prepare_data(args, idx, seq_data, seq_mask, datas, data_3d, image_data, device)
        if batch_data is None:
            continue
        x_list, preds = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            batch_mask_seq=None,
            gnn_batch_graph=batch_data['gnn_batch'],
            gnn_feature_batch=batch_data['features_batch'],
            batch_mask_gnn=None,
            graph_dict=batch_data['geo_gen'],
            node_id_all=batch_data['node_id_all'],
            edge_id_all=batch_data['edge_id_all'],
            img_batch=batch_data['image_batch']
        )
        # Calculated losses
        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, batch_data['targets'], batch_data['mask'])

        total_all_loss = all_loss.item() + total_all_loss
        total_lab_loss = lab_loss.item() + total_lab_loss
        total_cl_loss = cl_loss.item() + total_cl_loss
        batch_count += 1

        all_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        all_batch_preds.append(preds.detach())
        all_batch_labels.append(batch_data['targets'])

    all_predictions = torch.cat(all_batch_preds, dim=0)
    all_labels = torch.cat(all_batch_labels, dim=0)

    # Calculate average loss
    avg_all_loss = total_all_loss / batch_count if batch_count > 0 else float('inf')
    avg_lab_loss = total_lab_loss / batch_count if batch_count > 0 else float('inf')
    avg_cl_loss = total_cl_loss / batch_count if batch_count > 0 else float('inf')
    return avg_all_loss, avg_lab_loss, avg_cl_loss, all_predictions, all_labels


@torch.no_grad()
def val(args, model, scaler, val_idx_loader, seq_data, seq_mask, gnn_data, geo_data, image_data, device):
    model.eval()
    total_all_loss = 0
    total_lab_loss = 0
    total_cl_loss = 0
    batch_count = 0

    all_batch_preds = []
    all_batch_labels = []

    for idx in val_idx_loader:
        batch_data = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, image_data, device)
        if batch_data is None:
            continue

        x_list, preds = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            batch_mask_seq=None,
            gnn_batch_graph=batch_data['gnn_batch'],
            gnn_feature_batch=batch_data['features_batch'],
            batch_mask_gnn=None,
            graph_dict=batch_data['geo_gen'],
            node_id_all=batch_data['node_id_all'],
            edge_id_all=batch_data['edge_id_all'],
            img_batch=batch_data['image_batch']
        )

        if scaler is not None and args.task_type == 'reg':
            preds = torch.tensor(scaler.inverse_transform(preds.detach().cpu())).to(device)
            batch_data['targets'] = torch.tensor(scaler.inverse_transform(
                batch_data['targets'].detach().cpu())).to(device)

        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds,
                                                     batch_data['targets'], batch_data['mask'], args.cl_loss)

        total_all_loss = all_loss.item() + total_all_loss
        total_lab_loss = lab_loss.item() + total_lab_loss
        total_cl_loss = cl_loss.item() + total_cl_loss
        batch_count += 1

        all_batch_preds.append(preds.detach())
        all_batch_labels.append(batch_data['targets'])

    all_predictions = torch.cat(all_batch_preds, dim=0)
    all_labels = torch.cat(all_batch_labels, dim=0)

    y_true = all_labels.detach().cpu().numpy()
    y_pred = all_predictions.detach().cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    avg_all_loss = total_all_loss / batch_count if batch_count > 0 else float('inf')
    avg_lab_loss = total_lab_loss / batch_count if batch_count > 0 else float('inf')
    avg_cl_loss = total_cl_loss / batch_count if batch_count > 0 else float('inf')

    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']

    return result, avg_all_loss, avg_lab_loss, avg_cl_loss, all_predictions, all_labels


@torch.no_grad()
def test(args, model, scaler, test_idx_loader, seq_data, seq_mask, gnn_data, geo_data, image_data, device):
    y_true = []
    y_pred = []
    for idx in test_idx_loader:
        batch_data = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, image_data, device)
        if batch_data is None:
            continue

        x_list, preds = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            batch_mask_seq=None,
            gnn_batch_graph=batch_data['gnn_batch'],
            gnn_feature_batch=batch_data['features_batch'],
            batch_mask_gnn=None,
            graph_dict=batch_data['geo_gen'],
            node_id_all=batch_data['node_id_all'],
            edge_id_all=batch_data['edge_id_all'],
            img_batch=batch_data['image_batch']
        )
        if scaler is not None and args.task_type == 'reg':
            preds = torch.tensor(scaler.inverse_transform(preds.detach().cpu()).astype(np.float64))

        mask = batch_data['mask'].bool()
        y_true.append(batch_data['targets'][mask])
        y_pred.append(preds[mask])

    y_true = torch.cat(y_true, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if args.task_type == 'class':
        result = eval_rocauc(input_dict)['rocauc']
    else:
        result = eval_rmse(input_dict)['rmse']
    return result


def setup_logger(logs_file):
    logger = logging.getLogger()
    handler = logging.FileHandler(logs_file)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def generate_random_seed():
    random_seed = random.randint(0, 10000)
    print(f"Generated random seed: {random_seed}")
    return random_seed


def main(args):
    global smiles, image_filenames, image_labels, save_model, seed
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up log directories and files
    logs_dir = f'./LOG/{args.dataset}/{args.lr}_{args.epochs}_{args.batch_size}_{args.fusion}/'
    os.makedirs(logs_dir, exist_ok=True)
    logs_file = os.path.join(logs_dir, f"{current_time}.log")
    logger = setup_logger(logs_file)
    logger.info(f"Starting training with parameters: {vars(args)}")

    if args.fixed_seed:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Using fixed seed: {seed}")

    best_overall_result = None
    best_overall_test = None
    best_overall_epoch = 0
    all_test_results = []

    train_losses = []
    val_losses = []
    all_predictions = []
    all_labels = []

    for run in range(args.num_runs):
        if not args.fixed_seed:
            if not args.fixed_seed:
                seed = generate_random_seed()
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
        logger.info(f"-----------------------Run {run + 1}/{args.num_runs}-----------------------------")
        logger.info(f"seed:[{seed}]")

        # Device initialization
        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.cuda else 'cpu')
        torch.cuda.empty_cache()
        logger.info(f"Device set to: {device}")

        # Record hyperparameters
        hyperparams_log = (f"lr: {args.lr}, cl_loss: {args.cl_loss}, cl_loss_num: {args.cl_loss_num}, "
                           f"pro_num: {args.pro_num}, pool_type: {args.pool_type}, "
                           f"gnn_hidden_dim: {args.gnn_hidden_dim}, batch_size: {args.batch_size}, "
                           f"norm: {args.norm}, fusion: {args.fusion}")
        logger.info(hyperparams_log)

        # Data loaded
        data_path = f'data/{args.dataset}/{args.dataset}.csv'
        datas, args.seq_len = get_data(path=data_path, args=args)

        args.output_dim = args.num_tasks = datas.num_tasks()

        # Processing the SMILES sequence
        logger.info("Processing SMILES sequences...")
        smiles = datas.smiles()
        processor = MolecularDataProcessor(args)
        input_ids, attention_mask = processor.process_sequence_batch(smiles)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        seq_data = input_ids
        seq_mask = attention_mask

        # Processing 2D map data
        logger.info("Processing 2D molecular data...")
        args.gnn_atom_dim = get_atom_fdim()
        args.gnn_bond_dim = get_bond_fdim()

        logger.info(f"Atom feature dimension: {args.gnn_atom_dim}")
        logger.info(f"Bond feature dimension: {args.gnn_bond_dim}")

        mol_graphs = []
        for smiles in datas.smiles():
            mol_graph = MolGraph(smiles, args)
            mol_graphs.append(mol_graph)

        # Processing 3D geometry data
        logger.info("Loading 3D molecular data...")
        npz_data_path = os.path.join('data', args.dataset, 'part-000000.npz')
        try:
            data_processor = DataProcessor(args, device)

            if args.process_3d or not data_processor.check_processed_data():
                print("Processing 3D data...")
                data_3d = data_processor.process_3d_data()
                data_3d.get_data(device)
            else:
                print("Loading processed 3D data...")
                data_3d = InMemoryDataset(
                    npz_data_path=os.path.join(npz_data_path)
                )
                data_3d.get_data(device)
        except Exception as e:
            logger.error(f"Error processing 3D data:{str(e)}")
            raise

        # Processing image data
        logger.info("Processing image data...")
        smiles = datas.smiles()
        # Generate molecular images
        load_or_generate_images(args, smiles)
        # Generate index files
        process_dataset(args)
        # Generate label files
        generate_labels_file(args)

        image_folder = f'./data/{args.dataset}/image'
        image_labels_csv = f'./data/{args.dataset}/processed/{args.dataset}_label.csv'
        image_filenames, image_labels = load_filenames_and_labels(args, image_folder, image_labels_csv)

        image_transform = transforms.Compose([
            transforms.Resize((IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGEMOL_CONFIG['mean'],
                std=IMAGEMOL_CONFIG['std']
            )
        ])

        image_data = ImageDataset(
            filenames=image_filenames,
            labels=image_labels,
            img_transformer=image_transform,
            args=args
        )
 
        # Data segmentation
        train_data, val_data, test_data = split_data(data=datas, split_type=args.split_type, sizes=args.split_sizes,
                                                     seed=seed, args=args)
        train_idx = [data.idx for data in train_data]
        val_idx = [data.idx for data in val_data]
        test_idx = [data.idx for data in test_data]

        train_sampler = RandomSampler(train_idx)

        val_sampler = BatchSampler(val_idx, batch_size=args.batch_size, drop_last=True)
        test_sampler = BatchSampler(test_idx, batch_size=args.batch_size, drop_last=False)
        train_idx_loader = DataLoader(train_idx, batch_size=args.batch_size, sampler=train_sampler)

        # Task information processing
        if args.task_type == 'class':
            class_sizes = get_class_sizes(datas)
            for i, task_class_sizes in enumerate(class_sizes):
                print(f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
        elif args.task_type == 'reg':
            # Standardize data
            all_targets = datas.targets()
            scaler = StandardScaler().fit(all_targets)
            scaled_targets = scaler.transform(all_targets).tolist()

            for i, target in enumerate(scaled_targets):
                datas[i].set_targets(target)
        else:
            scaler = None

        model = Multi_modal(args, device)

        optimizer = Adam(params=model.parameters(), lr=args.init_lr, weight_decay=1e-5)
      
        # Cosine annealing learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.final_lr)

        ids = list(range(len(train_data)))
        print('Training model ...')
        scaler = None
        best_result = None
        best_test = None
        best_epoch = 0
        torch.backends.cudnn.enabled = False

        current_lr = optimizer.param_groups[0]['lr']

        for epoch in range(args.epochs):
            np.random.shuffle(ids)
            args.current_epoch = epoch

            # Training
            train_all_loss, train_label_loss, train_cl_loss, train_preds, train_labels = train(
                args, model, optimizer, train_idx_loader, seq_data, seq_mask, datas,
                data_3d, image_data, device
            )
            train_losses.append(train_all_loss)

            # Validation
            model.eval()
            val_result, val_all_loss, val_label_loss, val_cl_loss, val_preds, val_labels = val(
                args, model, scaler, val_sampler, seq_data,
                seq_mask, datas, data_3d, image_data, device
            )
            val_losses.append(val_all_loss)

            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

            all_predictions.extend(train_preds.cpu().numpy())
            all_labels.extend(train_labels.cpu().numpy())

            if best_result is None:
                best_result = val_result
                save_model = copy.deepcopy(model)
            else:
                if args.task_type == 'class':
                    if val_result > best_result:
                        best_result = val_result
                        save_model = copy.deepcopy(model)
                elif args.task_type == 'reg':
                    if val_result < best_result:
                        best_result = val_result
                        save_model = copy.deepcopy(model)

            # Test and record results
            result = test(args, save_model, scaler, test_sampler, seq_data, seq_mask, datas, data_3d, image_data,
                          device)

            if best_test is None:
                best_test = result
                best_epoch = epoch
            else:
                if (args.task_type == 'class' and result > best_test) or (
                        args.task_type == 'reg' and result < best_test):
                    best_test = result
                    best_epoch = epoch

            logger.info(
                f"Epoch: {epoch + 1}, lr: {current_lr:.6f}, train_all_loss: {train_all_loss:.4f}, "
                f"val_all_loss: {val_all_loss:.4f}, Best Test Result: {best_test:.4f}")

            torch.cuda.empty_cache()
        logger.info(f"Final Test Result: {best_test:.4f} at epoch {best_epoch + 1}")

        all_test_results.append(best_test)

        if best_overall_result is None:
            best_overall_result = best_result
            best_overall_test = best_test
            best_overall_epoch = best_epoch
        else:
            if (args.task_type == 'class' and best_overall_result < best_result) or \
                    (args.task_type == 'reg' and best_overall_result > best_result):
                best_overall_result = best_result
                best_overall_test = best_test
                best_overall_epoch = best_epoch

    # Calculate mean and standard deviation
    average_test_result = sum(all_test_results) / len(all_test_results)
    std_dev_test_result = statistics.stdev(all_test_results) if len(all_test_results) > 1 else 0.0

    # Output all test results
    logger.info("All Test Results: " + ", ".join(f"{result:.4f}" for result in all_test_results))
    # logger.info(f"Overall Best Test Result: {best_overall_test:.4f} at epoch {best_overall_epoch + 1} across all runs.")
    logger.info(f"Average Test Result across all runs: {average_test_result:.4f}")
    logger.info(f"Standard Deviation of Test Results: {std_dev_test_result:.4f}")

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    arg = get_args()
    main(arg)
