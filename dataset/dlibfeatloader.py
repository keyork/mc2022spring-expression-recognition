
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset


class DlibFeatSet(VisionDataset):

    def __init__(
            self,
            raw_data: torch.Tensor,
            train: bool = True,
    ):

        super(DlibFeatSet, self).__init__(root=None, transform=None, target_transform=None)
        self.data = raw_data[:, 1:]
        self.targets = raw_data[:, 0]
        self.train = train

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        return img, target

    def __len__(self):
        return len(self.data)


class DlibDataLoader:

    def __init__(self, raw_data, train, batch_size):

        self.raw_data = raw_data
        self.train = train
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None


    def _make_dataset(self):

        self.dataset = DlibFeatSet(self.raw_data, self.train)

    
    def _make_dataloader(self):

        self.dataloader = DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=8)
    

    def load_data(self):

        self._make_dataset()
        self._make_dataloader()