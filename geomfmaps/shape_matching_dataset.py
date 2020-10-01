# stdlib
from pathlib import Path
from itertools import permutations
from functools import partial
import hashlib
# 3p
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import __repr__
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.batch import SimpleBatch
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_points3d.utils.enums import ConvolutionFormat
from torch_points3d.utils.config import ConvolutionFormatFactory
# project
from utils import read_mesh


class ShapeMatchingDataset(InMemoryDataset):
    """Abstract class for shape matching dataset"""
    def __init__(self,
                 dataroot, transform=None, pre_transform=None,
                 neig=30, n_train=80, max_train=100, train=True):

        """Init dataset.
        Arguments:
            dataroot {string} -- Path to dataset
        Keyword Arguments:
            neig {int} -- number of eigenvectors used for representation (default: {30})
            transform {object} -- set of transforms to apply to dataset in training and testing (default: {None})
            pre_transform {object} -- set of transforms to apply to dataset before training starts (default: {None})
            n_train {int} -- Number of shapes used in training (default: {80})
            max_train {int} -- Total Number of shapes used in training & testing (default: {100})
            train {bool} -- set dataset to training mode (default: {True})
        """

        assert max_train >= n_train, f"max_train={max_train} is smaller than n_train={n_train}"
        # dataset path
        self.dataset_root = Path(dataroot)
        self.samples_path = (self.dataset_root / "off").resolve()
        self.spectral_path = (self.dataset_root / "spectral").resolve()
        self.processed_path = (self.dataset_root / "processed").resolve()
        self.raw_path = (self.dataset_root / "raw").resolve()
        self.raw_path.mkdir(parents=True, exist_ok=True)

        # params
        self.neig = neig
        self.train = train

        # train/test dataset
        self.sample_names = sorted([x for x in self.samples_path.iterdir() if x.is_file()])
        self.spectral_names = sorted([x for x in self.spectral_path.iterdir() if x.is_file()])
        self.corres_path = (self.dataset_root / "corres").resolve()

        # draw samples
        self.all_ind = list(range(len(self.sample_names)))
        if train:
            self.chosen_indices = self.all_ind[:n_train]
        else:
            self.chosen_indices = self.all_ind[n_train:max_train]

        # load data to ram (small dataset)
        self.used_names = sorted([self.sample_names[i].stem.split("_")[-1] for i in self.chosen_indices])
        self.split_name = '_'.join(self.used_names) + __repr__(pre_transform)
        self.split_name = hashlib.sha1(self.split_name.encode()).hexdigest()
        print(f"Using: {self.used_names}")
        if not train:
            self.samples = [self.load_sample(self.sample_names[i]) for i in tqdm(self.chosen_indices, desc="Loading samples")]

        self.evecs = [self.load_spectral(self.spectral_names[i])[1] for i in tqdm(self.chosen_indices, desc="Loading evecs")]
        self.combinations = list(permutations(range(len(self.chosen_indices)), 2))

        # load vts
        self.vts_names = sorted([x for x in self.corres_path.iterdir()
                                 if x.is_file() and not any(y in str(x) for y in ["sampleID", "sym"])])

        self.chosen_vts = [self.vts_names[i] for i in self.chosen_indices]
        self.vts = [np.loadtxt(v_path, dtype=np.int32) - 1 for v_path in tqdm(self.chosen_vts, desc="Loading vts")]

        super().__init__(dataroot, transform, pre_transform)

        self.data, self.slices = self.load_data(self.processed_path / f"{self.split_name}.pt")

    def load_data(self, path):
        '''This function is used twice to load data for both raw and pre_transformed
        '''
        data, slices = torch.load(path)

        return data, slices

    def load_sample(self, path):
        """Load and normalize a mesh."
        Arguments:
            path {string} -- path to mesh file.
        Returns:
            torch.Tensor -- Tensor containing vertices of shape. Size: `n_points x 3`
        """
        verts, _ = read_mesh(path)
        return torch.Tensor(verts)

    def load_spectral(self, path):
        """Load spectral data at `path`.
        The data is stored in a dict. This dict has the following keys:
            evals: eigen values. shape: neig x 1.
            evecs: eigen vectors. shape: `num_vertices` x neig.
            evecs_trans: transposed eigen vectors. shape: neig x `num_vertices`.
        Arguments:
            path {string} -- path to load spectral data from.
        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor) -- spectral data.
        """
        mat = sio.loadmat(path)
        return (torch.Tensor(mat['evals']).flatten()[:self.neig].float(),
                torch.Tensor(mat['evecs'])[:, :self.neig].float(),
                torch.Tensor(mat['evecs_trans'])[:self.neig, :].T.float())

    def load_c(self, i, j):
        """Compute functional map matrix from shape `i` to shape `j`.
        Arguments:
            i {int} -- index of source shape.
            j {int} -- index of target shape.
        Returns:
            torch.Tensor -- Tensor representing the functional map. Size: `n_eig x n_eig`.
        """
        # load eigen vectors & vts
        evec_i, evec_j = self.evecs[i], self.evecs[j]
        vts_i, vts_j = self.vts[i], self.vts[j]

        # compute C
        evec_i_a, evec_j_a = evec_i[vts_i], evec_j[vts_j]
        C_i_j = np.linalg.lstsq(evec_i_a, evec_j_a, rcond=None)[0]
        return torch.Tensor(C_i_j.T)

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, index):
        idx1, idx2 = self.combinations[index]

        # load pointcloud
        sample_x, sample_y = self.get(idx1), self.get(idx2)
        sample_x = sample_x if self.transform is None else self.transform(sample_x)
        sample_y = sample_y if self.transform is None else self.transform(sample_y)
        # load ground truth functional map
        C_gt = self.load_c(idx1, idx2)

        # continue with data class
        sample_x.C_gt = C_gt.unsqueeze(0)

        return sample_x, sample_y

    def _process(self):
        if (self.processed_path / f"{self.split_name}.pt").is_file():  # pragma: no cover
            return

        print('Processing...')

        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.process()

        path = self.processed_path / 'pre_transform.pt'
        torch.save(__repr__(self.pre_transform), path)

        print('Done!')

    def process(self):
        data_raw_list, data_list = self._process_filenames()

        self._save_data_list(data_list, self.processed_path / f"{self.split_name}.pt")
        self._save_data_list(data_raw_list, self.raw_path / f"{self.split_name}.pt", save_bool=len(data_raw_list) > 0)

    def _process_filenames(self):
        data_raw_list = []
        data_list = []

        has_pre_transform = self.pre_transform is not None

        id_scan = -1
        for idx in tqdm(self.chosen_indices):
            id_scan += 1
            pos = self.load_sample(self.sample_names[idx])
            evals_x, evecs_x, evecs_trans_x = self.load_spectral(self.spectral_names[idx])
            x = None
            id_scan_tensor = torch.from_numpy(np.asarray([id_scan])).clone()
            data = Data(pos=pos, x=x, evals_x=evals_x, evecs_x=evecs_x, evecs_trans_x=evecs_trans_x, id_scan=id_scan_tensor)
            data = SaveOriginalPosId()(data)
            data_raw_list.append(data.clone() if has_pre_transform else data)
            if has_pre_transform:
                data = self.pre_transform(data)
                data.nv = data.pos.shape[0]  # number of vertices
                data_list.append(data)
        if not has_pre_transform:
            return [], data_raw_list
        return data_raw_list, data_list

    def _save_data_list(self, datas, path_to_datas, save_bool=True):
        if save_bool:
            torch.save(self.collate(datas), path_to_datas)


class ShapeMatchingDatasetWrapper(BaseDataset):
    """ Wrapper around ShapeNet that creates shape matching datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - dataroot
            - pre_transforms
            - train_transforms
            - test_transforms
    """

    def __init__(self, hparams, train):
        """Init shape matching wrapper.
        Args:
            hparams (omegaconf): hydra dataset config
            train (bool): indicates if this is a training dataset
        """

        self.train = train
        self.use_data_class = True
        self.pre_transform = None
        self.train_transform, self.test_transform = None, None

        hparams.pre_transforms = hparams.pre_transforms
        hparams.train_transforms = hparams.train_transforms
        hparams.test_transforms = hparams.test_transforms

        params = OmegaConf.create(hparams)
        super().__init__(params)

        transform = self.train_transform if train else self.test_transform

        self._dataset = ShapeMatchingDataset(
            hparams.dataroot, pre_transform=self.pre_transform, transform=transform,
            neig=hparams.neig, n_train=hparams.n_train, max_train=hparams.max_train,
            train=train
        )

        self._test_loaders = []

    def get_dataloader(self, model, batch_size, shuffle, num_workers, precompute_multi_scale=True):
        if not self.use_data_class:
            return torch.utils.data.DataLoader(self._dataset, batch_size=batch_size,
                                               shuffle=shuffle and not self.train_sampler, num_workers=num_workers)
        conv_type = model.conv_type
        self._batch_size = batch_size

        batch_collate_function = self.__class__._get_collate_function(conv_type, precompute_multi_scale)
        dataloader = partial(
            torch.utils.data.DataLoader, collate_fn=batch_collate_function, worker_init_fn=lambda _: np.random.seed()
        )

        self._dataloader = dataloader(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle and not self.train_sampler,
            num_workers=num_workers,
            sampler=self.train_sampler,
        )

        if precompute_multi_scale:  # check if this excuted
            self.set_strategies(model)

        return self._dataloader

    @staticmethod
    def _get_collate_function(conv_type, is_multiscale):
        if is_multiscale:
            if conv_type.lower() == ConvolutionFormat.PARTIAL_DENSE.value.lower():
                return lambda datalist: MultiScaleBatch.from_data_list([y for x in datalist for y in x])
            else:
                raise NotImplementedError(
                    "MultiscaleTransform is activated and supported only for partial_dense format"
                )

        is_dense = ConvolutionFormatFactory.check_is_dense_format(conv_type)
        if is_dense:
            return lambda datalist: SimpleBatch.from_data_list(datalist)
        else:
            return lambda datalist: torch_geometric.data.batch.Batch.from_data_list(datalist)
