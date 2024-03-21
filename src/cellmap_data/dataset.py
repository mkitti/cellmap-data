# %%
import csv
from typing import Dict, Optional
from torch.utils.data import Dataset
import tensorstore as tswift
from fibsem_tools import read, read_xarray


# %%
class CellMapDataset(Dataset):
    def __init__(
        self, dataset_dict: Optional[Dict[str, Dict[str, str | float]]] = None
    ):
        self.dataset_dict = dataset_dict
        self.construct()

    def __len__(self): ...

    def __getitem__(self, idx): ...

    def from_csv(self, csv_path):
        # Load file data from csv file
        dataset_dict = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in dataset_dict:
                    dataset_dict[row[0]] = {}
                dataset_dict[row[0]]["raw"] = row[1]
                dataset_dict[row[0]]["gt"] = row[2]
                if len(row) > 3:
                    dataset_dict[row[0]]["weight"] = row[3]
                else:
                    dataset_dict[row[0]]["weight"] = 1.0

        self.dataset_dict = dataset_dict
        self.construct()

    def construct(self): ...


# %%
