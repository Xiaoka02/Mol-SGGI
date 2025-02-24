from utils.dataset import InMemoryDataset, get_data
from preprocess.gem_featurizer import GeoPredTransformFn
import json
import os


class DataProcessor:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def process_3d_data(self):
        """Processing 3D data"""
        print("\nStart processing 3D data...")

        data_path = os.path.join('data', self.args.dataset, f'{self.args.dataset}.csv')
        datas, self.args.seq_len = get_data(path=data_path, args=self.args)

        compound_encoder_config = self._load_json_config(self.args.compound_encoder_config)
        model_config = self._load_json_config(self.args.model_config)

        if self.args.dropout_rate is not None:
            compound_encoder_config['dropout_rate'] = self.args.dropout_rate
            model_config['dropout_rate'] = self.args.dropout_rate

        data_3d = InMemoryDataset(datas.smiles())
        transform_fn = GeoPredTransformFn(
            model_config['pretrain_tasks'],
            model_config['mask_ratio']
        )

        print("Start converting 3D data...")
        data_3d.transform(transform_fn, num_workers=1)

        save_dir = os.path.join('data', self.args.dataset)
        os.makedirs(save_dir, exist_ok=True)
        data_3d.save_data(save_dir)

        print("3D data processing complete!")
        return data_3d

    def _load_json_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configuration file {path} :{str(e)}")
            raise

    def check_processed_data(self):
        """Checking the existence of processed data"""
        npz_path = os.path.join('data', self.args.dataset, 'part-000000.npz')
        return os.path.exists(npz_path)
