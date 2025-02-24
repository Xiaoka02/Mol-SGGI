import os
import logging
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from torchvision import transforms
from tqdm import tqdm

IMAGEMOL_CONFIG = {
    'size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'model': 'resnet50'
}


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, index=None, img_transformer=None, normalize=None, ret_index=False, args=None):
        """
        Args:
            filenames: list of image file paths
            labels: list of labels
            index: list of indexes
            img_transformer: image transformer
            normalize: standardization function
            ret_index: whether to return the index
            args: Argument object
        """
        super().__init__()
        self.args = args
        self.filenames = filenames
        self.labels = labels
        self.total = len(self.filenames)
        self.normalize = normalize
        self._image_transformer = img_transformer
        self.ret_index = ret_index

        self.transform = transforms.Compose([
            transforms.Resize((IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGEMOL_CONFIG['mean'],
                std=IMAGEMOL_CONFIG['std']
            )
        ])

        if not all(os.path.exists(f) for f in filenames):
            missing_files = [f for f in filenames if not os.path.exists(f)]
            raise FileNotFoundError(f"Missing image files: {missing_files[:5]}...")

        if index is not None:
            self.index = index
        else:
            self.index = [os.path.splitext(os.path.split(f)[1])[0] for f in filenames]

    def get_image(self, index):
        """
        Args:
            index: image index
        Returns:
            Processed image tensor
        """
        try:
            filename = self.filenames[index]
            img = Image.open(filename).convert('RGB')
            if self._image_transformer:
                img = self._image_transformer(img)
            return img
        except Exception as e:
            logging.error(f"Error loading image {filename}: {e}")
            blank_img = Image.new('RGB', (IMAGEMOL_CONFIG['size'], IMAGEMOL_CONFIG['size']), 'white')
            return self._image_transformer(blank_img) if self._image_transformer else blank_img

    def __getitem__(self, index):
        data = self.get_image(index)
        if self.normalize is not None:
            data = self.normalize(data)
        if self.ret_index:
            return data, self.labels[index], self.index[index]
        return data, self.labels[index]

    def __len__(self):
        """Returns the size of the dataset"""
        return self.total


def Smiles2Img(smiles, size=(224, 224), savePath=None, quality=95):
    """
    Converting SMILES to molecular images
    Args:
        smiles: SMILES string
        size: image size
        savePath: save path
        quality: image quality
    Returns:
         PIL.Image object or None
    """
    try:
        if isinstance(size, int):
            size = (size, size)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        AllChem.Compute2DCoords(mol)

        img = Draw.MolToImage(
            mol,
            size=size,
            kekulize=True,
            wedgeBonds=True,
            fitImage=True
        )

        if savePath:
            img.save(savePath, quality=quality)

        return img

    except Exception as e:
        logging.error(f"Error generating image for SMILES {smiles}: {e}")
        return None


def load_or_generate_images(args, smiles):
    """
    Load or generate molecular images
    """
    dataset_path = os.path.join('./data', args.dataset)
    image_folder = os.path.join(dataset_path, 'image')

    os.makedirs(image_folder, exist_ok=True)

    regenerate_images = getattr(args, 'regenerate_images', False)
    image_size = getattr(args, 'image_size', IMAGEMOL_CONFIG['size'])
    image_quality = getattr(args, 'image_quality', 95)

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    generated_count = 0
    failed_count = 0

    for idx, smi in enumerate(tqdm(smiles, desc="Processing molecules")):
        save_path = os.path.join(image_folder, f'{idx + 1}.png')

        if not os.path.exists(save_path) or regenerate_images:
            try:
                img = Smiles2Img(
                    smi,
                    size=image_size,
                    savePath=save_path,
                    quality=image_quality
                )
                if img is not None:
                    generated_count += 1
                else:
                    failed_count += 1
                    logging.warning(f"Failed to generate image for SMILES: {smi}")
            except Exception as e:
                failed_count += 1
                logging.error(f"Error generating image for SMILES {smi}: {e}")

    logging.info(f"Generated {generated_count} images, failed {failed_count} images")
    return generated_count, failed_count


def load_filenames_and_labels(args, image_folder, image_labels_csv):
    assert args.task_type in ["class", "reg"], f"Unsupported task type: {args.task_type}"

    try:
        if not os.path.exists(image_labels_csv):
            raise FileNotFoundError(f"Labels file not found: {image_labels_csv}")
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        df = pd.read_csv(image_labels_csv)

        if "filename" not in df.columns:
            raise ValueError("CSV file must contain 'filename' column")

        filenames = df["filename"].values

        label_columns = [col for col in df.columns if col != "filename"]
        if not label_columns:
            raise ValueError("No label columns found in CSV file")

        labels = df[label_columns].values

        if args.task_type == "class":
            labels = labels.astype(int)
        else:  # reg
            labels = labels.astype(float)

        full_paths = [os.path.join(image_folder, filename) for filename in filenames]

        missing_files = [f for f in full_paths if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing image files: {missing_files[:5]}...")

        logging.info(f"Successfully loaded {len(full_paths)} images with {labels.shape[1]} labels")
        return full_paths, labels

    except Exception as e:
        logging.error(f"Error in load_filenames_and_labels: {e}")
        raise


def generate_labels_file(args):
    processed_file_path = os.path.join(f'data/{args.dataset}/processed/{args.dataset}_processed.csv')
    dataset_path = os.path.join(f'data/{args.dataset}/processed')
    labels_file_path = os.path.join(dataset_path, f'{args.dataset}_label.csv')

    try:
        if not os.path.exists(processed_file_path):
            raise FileNotFoundError(f"Processed file not found: {processed_file_path}")

        df = pd.read_csv(processed_file_path)

        image_filenames = [f"{i + 1}.png" for i in range(len(df))]

        label_cols = [col for col in df.columns if col not in ['smiles', 'index']]

        labels_df = pd.DataFrame({
            "filename": image_filenames,
            **{col: df[col].values for col in label_cols}
        })

        os.makedirs(os.path.dirname(labels_file_path), exist_ok=True)
        labels_df.to_csv(labels_file_path, index=False)
        logging.info(f"Successfully generated labels file: {labels_file_path}")

    except Exception as e:
        logging.error(f"Error generating labels file: {e}")
        raise


def process_dataset(args):
    raw_dataset = os.path.join("data", args.dataset, f'{args.dataset}.csv')
    save_dir = os.path.join("data", args.dataset, "processed")
    save_dataset = os.path.join(save_dir, f'{args.dataset}_processed.csv')

    try:
        if not os.path.exists(raw_dataset):
            raise FileNotFoundError(f"Raw dataset not found: {raw_dataset}")

        df = pd.read_csv(raw_dataset)

        df.insert(0, "index", range(1, len(df) + 1))

        os.makedirs(save_dir, exist_ok=True)

        df.to_csv(save_dataset, index=False)
        logging.info(f"Successfully processed dataset: {save_dataset}")

    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise
