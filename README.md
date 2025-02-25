# Mol-SGGI
A Comprehensive Multi-Representation Learning Framework for Molecular Property Prediction

## OverView<br>

![Mol-SGGI framework diagram](image/Mol-SGGI%20framework%20diagram.png)

## Dataset
We used MoleculeNet as our benchmark test, and the experimental MoleculeNet dataset is available under [this link](https://moleculenet.org/datasets-1).

## Setup Environment
- `python==3.9`
- `torch==2.2.0+cu118`
- `torch-cluster==1.6.3+pt22cu118`
- `torch-scatter==2.1.2+pt22cu118`
- `torch-sparse==0.6.18+pt22cu118`
- `torch-spline-conv==1.2.2+pt22cu118`
- `torch-geometric==2.5.3`
- `paddlepaddle==2.4.0`
- `paddlepaddle-gpu==2.6.1`
- `pandas==1.3.5`
- `pgl==2.2.3.post0`
- `pillow==10.2.0`
- `rdkit==2024.3.5`
- `scikit-learn==1.0.2`
- `scipy==1.7.3`

## Usage
Run the command directly and specify the dataset name and task type. <br>

`python main.py --dataset {dataset_name} --task_type {reg/class}`

If the specified dataset does not exist or is unprocessed, the program automatically performs the data processing steps and generates molecular characterization data for the corresponding dataset. The data will be stored in the following structure:
```bash
data/
  ├── bace/
      ├── image/
      ├── processed/
          └── bace_label.csv
          └── bace_processed.csv
      └── bace.csv
      └── part-000000.npz
