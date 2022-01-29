import numpy as np
import torch
from skimage.io import imread
from PIL import Image
from torch.utils import data
from tqdm import tqdm
import random



class SegmentationDataSet2(data.Dataset):
    """Image segmentation dataset (tasks1&2) with caching and pretransforms."""

    def __init__(
        self,
        inputs: list,
        targets: list,
        n_classes: int,
        transform=None,
        augmenter=None,
        use_cache=False,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.resize=resize
        self.new_size = new_size

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = Image.open(str(img_name)), Image.open(str(tar_name))

                # resizing
                if self.resize:
                    img = img.resize(self.new_size, resample=Image.LANCZOS)
                    tar = tar.resize(self.new_size, resample=Image.LANCZOS)

                # convert to numpy
                img, tar = np.array(img), np.array(tar)
                if self.n_classes > 1:
                    tar[tar == 100] = self.n_classes - 1

                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = Image.open(str(input_ID)), Image.open(str(target_ID))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                y = y.resize(self.new_size, resample=Image.LANCZOS)

            # convert to numpy
            x, y = np.array(x), np.array(y)
            if self.n_classes > 1:
                y[y == 100] = self.n_classes - 1

        if self.augmenter is not None:
            augmented = self.augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y
        
class SegmentationDataSet2Test(data.Dataset):
    """Image segmentation dataset (tasks1&2) with caching and pretransforms."""

    def __init__(
        self,
        inputs: list,
        n_classes: int,
        transform=None,
        use_cache=False,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
    ):
        self.inputs = inputs
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.resize=resize
        self.new_size = new_size

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name in zip(progressbar, self.inputs):
                img = Image.open(str(img_name))

                # resizing
                if self.resize:
                    img = img.resize(self.new_size, resample=Image.LANCZOS)

                # convert to numpy
                img = np.array(img)

                if self.pre_transform is not None:
                    img = self.pre_transform(img)

                self.cached_data.append((img))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]

            # Load input and target
            x = Image.open(str(input_ID))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)

            # convert to numpy
            x = np.array(x)

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)

        # Typecasting
        x = torch.from_numpy(x).type(self.inputs_dtype)
        

        return x


class SegmentationDataSet3(data.Dataset):
    """Image segmentation dataset (tasks1&2) with caching, pretransforms and multiprocessing."""

    def __init__(
        self,
        inputs: list,
        targets: list,
        n_classes: int,
        transform=None,
        use_cache=False,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
        processes: int=6,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.resize=resize
        self.new_size = new_size
        self.processes = processes

        if self.use_cache:
            from itertools import repeat
            from multiprocessing import Pool

            with Pool(processes=self.processes) as pool:
                self.cached_data = pool.starmap(
                    self.read_images, zip(inputs, targets, repeat(self.pre_transform), repeat(self.n_classes), repeat(self.resize), repeat(self.new_size))
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = Image.open(str(input_ID)), Image.open(str(target_ID))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                y = y.resize(self.new_size, resample=Image.LANCZOS)

            # convert to numpy
            x, y = np.array(x), np.array(y)
            if self.n_classes > 1:
                y[y == 100] = self.n_classes - 1

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

    @staticmethod
    def read_images(inp, tar, pre_transform, n_classes: int, resize, new_size):

        # Load input and target
        inp, tar = Image.open(str(inp)), Image.open(str(tar))

        # resizing
        if resize:
            inp=inp.resize(new_size,resample=Image.LANCZOS)
            tar=tar.resize(new_size,resample=Image.LANCZOS)

        #convert to numpy
        inp, tar = np.array(inp), np.array(tar)
        if n_classes > 1:
            tar[tar==100] = n_classes-1

        if pre_transform:
            inp, tar = pre_transform(inp, tar)

        return inp, tar

class SegmentationDataSet3_aug(data.Dataset):
    """Image segmentation dataset (tasks1&2) with augmentation, caching, pretransforms and multiprocessing."""

    def __init__(
        self,
        inputs: list,
        targets: list,
        n_classes: int,
        transform=None,
        augmenter=None,
        use_cache=False,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
        processes: int=6,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.resize=resize
        self.new_size = new_size
        self.processes = processes

        if self.use_cache:
            from itertools import repeat
            from multiprocessing import Pool

            with Pool(processes=self.processes) as pool:
                self.cached_data = pool.starmap(
                    self.read_images, zip(inputs, targets, repeat(self.pre_transform), repeat(self.n_classes), repeat(self.resize), repeat(self.new_size))
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = Image.open(str(input_ID)), Image.open(str(target_ID))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                y = y.resize(self.new_size, resample=Image.LANCZOS)

            # convert to numpy
            x, y = np.array(x), np.array(y)
            if self.n_classes > 1:
                y[y == 100] = self.n_classes - 1

        if self.augmenter is not None:
            augmented = self.augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

    @staticmethod
    def read_images(inp, tar, pre_transform, n_classes: int, resize, new_size):

        # Load input and target
        inp, tar = Image.open(str(inp)), Image.open(str(tar))

        # resizing
        if resize:
            inp=inp.resize(new_size,resample=Image.LANCZOS)
            tar=tar.resize(new_size,resample=Image.LANCZOS)

        #convert to numpy
        inp, tar = np.array(inp), np.array(tar)
        if n_classes > 1:
            tar[tar==100] = n_classes-1

        if pre_transform:
            inp, tar = pre_transform(inp, tar)

        return inp, tar


class DataSetTask3SP(data.Dataset):
    """Image segmentation dataset task 3 super-resolution end-to-end with caching and pretransforms."""

    def __init__(
        self,
        inputs: list,
        targets: dict,
        transform=None,
        augmenter=None,
        use_cache=False,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
    ):

        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.resize = resize
        self.new_size = new_size


        if self.use_cache:
            self.cached_data = []
            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                # Load input and target
                img = Image.open(str(img_name))
                tar_list = []
                for target in tar_name:
                    tar_list.append(Image.open(str(target)))

                # resizing
                if self.resize:
                    img = img.resize(self.new_size, resample=Image.LANCZOS)
                    tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                # convert to numpy
                img = np.array(img)
                tar_list = [np.array(target) for target in tar_list]
                # if n_classes > 1:
                #     target[target == 100] = n_classes - 1
                tar = np.array(tar_list)
                tar = np.moveaxis(tar, 0, -1)

                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x = Image.open(str(input_ID))
            tar_list = []
            for target in target_ID:
                tar_list.append(Image.open(str(target)))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

            # convert to numpy
            x = np.array(x)
            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            y = np.array(tar_list)
            y = np.moveaxis(y, 0, -1)

        if self.augmenter is not None:
            # self.augmenter.ignore_channels = random.sample(range(3), 2)
            augmented = self.augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

class DataSetTask3CropSP(data.Dataset):
    """Image segmentation dataset for task 3 using random crop method with with caching and pretransforms."""

    def __init__(
        self,
        inputs: list,
        targets: dict,
        n_classes: int,
        transform=None,
        augmenter=None,
        use_cache=False,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
    ):

        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.resize = resize
        self.new_size = new_size


        if self.use_cache:
            self.cached_data = []
            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                # Load input and target
                img = Image.open(str(img_name))
                tar_list = []
                for target in tar_name:
                    tar_list.append(Image.open(str(target)))

                # resizing
                if self.resize:
                    img = img.resize(self.new_size, resample=Image.LANCZOS)
                    tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                # convert to numpy
                img = np.array(img)
                tar_list = [np.array(target) for target in tar_list]
                # if n_classes > 1:
                #     target[target == 100] = n_classes - 1
                tar = np.array(tar_list)
                tar = np.moveaxis(tar, 0, -1)

                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x = Image.open(str(input_ID))
            tar_list = []
            for target in target_ID:
                tar_list.append(Image.open(str(target)))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

            # convert to numpy
            x = np.array(x)
            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            y = np.array(tar_list)
            y = np.moveaxis(y, 0, -1)

        if self.augmenter is not None:
            self.augmenter.ignore_channels = random.sample(range(3), 2)
            augmented = self.augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y


class DataSetTask3Crop(data.Dataset):
    """Image segmentation dataset for task 3 using random crop method with caching, pretransforms and multiprocessing."""

    def __init__(
            self,
            inputs: list,
            targets: dict,
            n_classes: int,
            transform=None,
            augmenter=None,
            use_cache=False,
            pre_transform=None,
            resize=True,
            new_size=(512, 288),
            processes: int = 6,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.resize = resize
        self.new_size = new_size
        self.processes = processes

        if self.use_cache:
            from itertools import repeat
            from multiprocessing import Pool

            with Pool(processes=self.processes) as pool:
                self.cached_data = pool.starmap(
                    self.read_images,
                    zip(inputs, targets, repeat(self.pre_transform), repeat(self.n_classes), repeat(self.resize),
                        repeat(self.new_size))
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x = Image.open(str(input_ID))
            tar_list = []
            for target in target_ID:
                tar_list.append(Image.open(str(target)))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

            # convert to numpy
            x = np.array(x)
            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            y = np.array(tar_list)
            y = np.moveaxis(y, 0, -1)

        if self.augmenter is not None:
            self.augmenter.ignore_channels = random.sample(range(3), 2)
            augmented = self.augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

    @staticmethod
    def read_images(inp, targets, pre_transform, n_classes: int, resize, new_size):

        # Load input and target
        inp = Image.open(str(inp))
        tar_list = []
        for target in targets:
            tar_list.append(Image.open(str(target)))

        # resizing
        if resize:
            inp = inp.resize(new_size, resample=Image.LANCZOS)
            tar_list = [target.resize(new_size, resample=Image.LANCZOS) for target in tar_list]

        # convert to numpy
        inp = np.array(inp)
        tar_list = [np.array(target) for target in tar_list]
        # if n_classes > 1:
        #     target[target == 100] = n_classes - 1
        tar = np.array(tar_list)
        tar = np.moveaxis(tar, 0, -1)

        if pre_transform:
            inp, tar = pre_transform(inp, tar)

        return inp, tar



class DataSetTask3PatchSP(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""

    def __init__(
        self,
        inputs: list,
        targets: dict,
        n_classes: int,
        transform=None,
        augmenter=None,
        use_cache=False,
        pre_transform=None,
        crop_height=270,
        crop_width=480,
        processes: int = 6,
    ):

        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.processes = processes
        self.crop_height = crop_height
        self.crop_width = crop_width

        if self.use_cache:
            self.cached_data = []
            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                # Load input and target
                img = Image.open(str(img_name))
                tar_list = []
                for target in tar_name:
                    tar_list.append(Image.open(str(target)))

                # resizing
                if self.resize:
                    img = img.resize(self.new_size, resample=Image.LANCZOS)
                    tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                # convert to numpy
                img = np.array(img)
                tar_list = [np.array(target) for target in tar_list]
                # if n_classes > 1:
                #     target[target == 100] = n_classes - 1
                tar = np.array(tar_list)
                tar = np.moveaxis(tar, 0, -1)

                self.cached_data.append((img, tar))

    def blockshaped(self, arr):
        """
        Return an array of shape (n, self.crop_height, self.crop_width, channels) where
        n * self.crop_height * self.crop_width *channels = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w, channels = arr.shape
        assert h % self.crop_height == 0, f"{h} rows is not evenly divisible by {self.crop_height}"
        assert w % self.crop_width == 0, f"{w} cols is not evenly divisible by {self.crop_width}"
        return (arr.reshape(h // self.crop_height, self.crop_height, w // self.crop_width, self.crop_width, channels)
                .swapaxes(1, 2)
                .reshape(-1, self.crop_height, self.crop_width, channels))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x = Image.open(str(input_ID))
            tar_list = []
            for target in target_ID:
                tar_list.append(Image.open(str(target)))

            # cropping

            # convert to numpy
            x = np.array(x)
            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            y = np.array(tar_list)
            y = np.moveaxis(y, 0, -1)


        # crop
        x = self.blockshaped(x)
        y = self.blockshaped(y)

        if self.augmenter is not None:
            x_b = []
            y_b = []
            for i in range(x.shape[0]):
                augmented = self.augmenter(image=x[i], mask=y[i])
                x_b.append(augmented['image']), y_b.append(augmented['mask'])
            x, y = np.array(x_b), np.array(y_b)


        # Preprocessing
        if self.transform is not None:
            x_b = []
            y_b = []
            for i in range(x.shape[0]):
                transform_output = self.transform(x[i], y[i])
                x_b.append(transform_output[0]), y_b.append(transform_output[1])
            x, y = np.array(x_b), np.array(y_b)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

class DataSetTask3Patch(data.Dataset):
    """Image segmentation dataset with caching, pretransforms and multiprocessing."""

    def __init__(
            self,
            inputs: list,
            targets: dict,
            n_classes: int,
            transform=None,
            augmenter=None,
            use_cache=False,
            pre_transform=None,
            crop_height=270,
            crop_width=480,
            processes: int = 6,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.processes = processes
        self.crop_height = crop_height
        self.crop_width = crop_width

        if self.use_cache:
            from itertools import repeat
            from multiprocessing import Pool

            with Pool(processes=self.processes) as pool:
                self.cached_data = pool.starmap(
                    self.read_images,
                    zip(inputs, targets, repeat(self.pre_transform), repeat(self.n_classes), repeat(self.crop_factor))
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x = Image.open(str(input_ID))
            tar_list = []
            for target in target_ID:
                tar_list.append(Image.open(str(target)))

            # cropping

            # convert to numpy
            x = np.array(x)
            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            y = np.array(tar_list)
            y = np.moveaxis(y, 0, -1)

            # crop
            x_b = self.blockshaped(x)
            y_b = self.blockshaped(y)

            x = []
            y = []

        if self.augmenter is not None:
            for i in range(x_b.shape[0]):
                augmented = self.augmenter(image=x_b[i], mask=y_b[i])
                x.append(augmented['image']), y.append(augmented['mask'])

            x_b, y_b = np.array(x), np.array(y)

            x = []
            y = []

        # Preprocessing
        if self.transform is not None:
            for i in range(x_b.shape[0]):
                transform_output = self.transform(x_b[i], y_b[i])
                x.append(transform_output[0]), y.append(transform_output[1])

        x, y = np.array(x), np.array(y)
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

    def blockshaped(self, arr):
        """
        Return an array of shape (n, self.crop_height, self.crop_width, channels) where
        n * self.crop_height * self.crop_width *channels = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w, channels = arr.shape
        assert h % self.crop_height == 0, f"{h} rows is not evenly divisible by {self.crop_height}"
        assert w % self.crop_width == 0, f"{w} cols is not evenly divisible by {self.crop_width}"
        return (arr.reshape(h // self.crop_height, self.crop_height, w // self.crop_width, self.crop_width, channels)
                .swapaxes(1, 2)
                .reshape(-1, self.crop_height, self.crop_width, channels))

    @staticmethod
    def read_images(inp, targets, pre_transform, n_classes: int, nrows, ncols):

        # Load input and target
        inp = Image.open(str(inp))
        tar_list = []
        for target in targets:
            tar_list.append(Image.open(str(target)))

        # convert to numpy
        inp = np.array(inp)
        tar_list = [np.array(target) for target in tar_list]
        # if n_classes > 1:
        #     target[target == 100] = n_classes - 1
        tar = np.array(tar_list)
        tar = np.moveaxis(tar, 0, -1)

        if pre_transform:
            inp, tar = pre_transform(inp, tar)

        h, w, channels = inp.shape
        inp = inp.reshape(h // nrows, nrows, w // ncols, ncols, channels).swapaxes(1, 2).reshape(-1, nrows, ncols,
                                                                                                 channels)

        h, w, channels = tar.shape
        tar = tar.reshape(h // nrows, nrows, w // ncols, ncols, channels).swapaxes(1, 2).reshape(-1, nrows, ncols,
                                                                                                 channels)

        return inp, tar

class DataSetTask3SuperSP(data.Dataset):
    """Image segmentation dataset for super-resolution models (DS and US) for task 3 with caching and pretransforms."""

    def __init__(
            self,
            inputs,
            is_mask: bool = False,
            transform=None,
            augmenter=None,
            use_cache=False,
            pre_transform=None,
            resize=True,
            new_size=(512, 288),
    ):
        self.inputs = inputs
        self.is_mask = is_mask
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.resize = resize
        self.new_size = new_size

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name in zip(progressbar, self.inputs,):
                if not self.is_mask:
                    img = Image.open(str(img_name))

                    # resizing
                    if self.resize:
                        img = img.resize(self.new_size, resample=Image.LANCZOS)

                    # convert to numpy
                    img = np.array(img)

                else:
                    tar_list = []
                    for target in img_name:
                        tar_list.append(Image.open(str(target)))

                    # resizing
                    if self.resize:
                        tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                    # convert to numpy
                    tar_list = [np.array(target) for target in tar_list]
                    # if n_classes > 1:
                    #     target[target == 100] = n_classes - 1
                    img = np.array(tar_list)
                    img = np.moveaxis(img, 0, -1)

                if self.pre_transform is not None:
                    img = self.pre_transform(img)

                self.cached_data.append((img))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]

            # Load input and target
            if not self.is_mask:
                x = Image.open(str(input_ID))

                # resizing
                if self.resize:
                    x = x.resize(self.new_size, resample=Image.LANCZOS)

                # convert to numpy
                x = np.array(x)

            else:
                tar_list = []
                for target in input_ID:
                    tar_list.append(Image.open(str(target)))

                # resizing
                if self.resize:
                    tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                # convert to numpy
                tar_list = [np.array(target) for target in tar_list]
                # if n_classes > 1:
                #     target[target == 100] = n_classes - 1
                x = np.array(tar_list)
                x = np.moveaxis(x, 0, -1)

        if self.augmenter is not None:
            # self.augmenter.ignore_channels = random.sample(range(3), 2)
            augmented = self.augmenter(image=x)
            x = augmented['image']

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)

        # Typecasting
        x = torch.from_numpy(x).type(self.inputs_dtype)
        # if self.is_mask:
        #     y = x.type(self.targets_dtype)
        # else:
        #     y = x

        return x, x


class DataSetTask3Super(data.Dataset):
    """Image segmentation dataset for super-resolution models (DS and US) for task 3  with caching, pretransforms and multiprocessing."""

    def __init__(
            self,
            inputs,
            is_mask: bool = False,
            transform=None,
            augmenter=None,
            use_cache=False,
            pre_transform=None,
            resize=True,
            new_size=(512, 288),
            processes: int = 6,
    ):
        self.inputs = inputs
        self.is_mask = is_mask
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.resize = resize
        self.new_size = new_size
        self.processes = processes

        if self.use_cache:
            from itertools import repeat
            from multiprocessing import Pool

            with Pool(processes=self.processes) as pool:
                self.cached_data = pool.starmap(
                    self.read_images,
                    zip(inputs, repeat(self.is_mask), repeat(self.pre_transform), repeat(self.resize),
                        repeat(self.new_size))
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]

            # Load input and target
            if not self.is_mask:
                x = Image.open(str(input_ID))

                # resizing
                if self.resize:
                    x = x.resize(self.new_size, resample=Image.LANCZOS)

                # convert to numpy
                x = np.array(x)

            else:
                tar_list = []
                for target in input_ID:
                    tar_list.append(Image.open(str(target)))

                # resizing
                if self.resize:
                    tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                # convert to numpy
                tar_list = [np.array(target) for target in tar_list]
                # if n_classes > 1:
                #     target[target == 100] = n_classes - 1
                x = np.array(tar_list)
                x = np.moveaxis(x, 0, -1)

        if self.augmenter is not None:
            # self.augmenter.ignore_channels = random.sample(range(3), 2)
            augmented = self.augmenter(image=x)
            x = augmented['image']

        # Preprocessing
        if self.transform is not None:
            x = self.transform(x)

        # Typecasting
        x = torch.from_numpy(x).type(self.inputs_dtype)
        # if self.is_mask:
        #     y = x.type(self.targets_dtype)
        # else:
        #     y = x

        return x, x

    @staticmethod
    def read_images(inp, is_mask, pre_transform, resize, new_size):

        # Load input and target
        if not is_mask:
            inp = Image.open(str(inp))

            # resizing
            if resize:
                inp = inp.resize(new_size, resample=Image.LANCZOS)
            # convert to numpy
            inp = np.array(inp)

        else:
            tar_list = []
            for target in inp:
                tar_list.append(Image.open(str(target)))

            # resizing
            if resize:
                tar_list = [target.resize(new_size, resample=Image.LANCZOS) for target in tar_list]

            # convert to numpy
            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            inp = np.array(tar_list)
            inp = np.moveaxis(inp, 0, -1)


        if pre_transform:
            inp = pre_transform(inp)

        return inp


class DataSetT3FullSP(data.Dataset):
    """Image segmentation dataset for the full super-resolution model for task 3 with caching and pretransforms."""

    def __init__(
        self,
        inputs: list,
        targets: dict,
        transform=None,
        augmenter=None,
        use_cache=False,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
    ):

        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.resize = resize
        self.new_size = new_size


        if self.use_cache:
            self.cached_data = []
            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                # Load input and target
                img = Image.open(str(img_name))
                tar_list = []
                for target in tar_name:
                    tar_list.append(Image.open(str(target)))

                # resizing
                if self.resize:
                    img = img.resize(self.new_size, resample=Image.LANCZOS)
                    tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                # convert to numpy
                img = np.array(img)

                tar_list = [np.array(target) for target in tar_list]
                # if n_classes > 1:
                #     target[target == 100] = n_classes - 1
                tar = np.array(tar_list)
                tar = np.moveaxis(tar, 0, -1)  # make sure you fix that in the transforms


                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x = Image.open(str(input_ID))
            tar_list = []
            for target in target_ID:
                tar_list.append(Image.open(str(target)))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

            # convert to numpy
            x = np.array(x)

            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            y = np.array(tar_list)
            y = np.moveaxis(y, 0, -1)


        if self.augmenter is not None:
            # self.augmenter.ignore_channels = random.sample(range(3), 2)
            augmented = self.augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y


class DataSetT3BnMlSP(data.Dataset):
    """Image segmentation dataset for bottleneck model in super resolution  for task 3 with caching and pretransforms."""
 

    def __init__(
        self,
        inputs: list,
        targets: dict,
        ds_input: torch.nn.Module,
        ds_target: torch.nn.Module,
        device='cpu',
        transform=None,
        augmenter=None,
        use_cache=False,
        input_transforms=None,
        target_transforms=None,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
    ):

        self.inputs = inputs
        self.targets = targets
        self.device = device
        self.ds_target = ds_target.to(self.device)
        self.ds_input = ds_input.to(self.device)
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.resize = resize
        self.new_size = new_size
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms


        self.ds_input.eval()
        self.ds_target.eval()


        if self.use_cache:
            self.cached_data = []
            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                # Load input and target
                img = Image.open(str(img_name))
                tar_list = []
                for target in tar_name:
                    tar_list.append(Image.open(str(target)))

                # resizing
                if self.resize:
                    img = img.resize(self.new_size, resample=Image.LANCZOS)
                    tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                # convert to numpy
                img = np.array(img)
                img = self.downsample(img, self.ds_input, self.inputs_dtype, self.device, self.input_transforms)

                tar_list = [np.array(target) for target in tar_list]
                # if n_classes > 1:
                #     target[target == 100] = n_classes - 1
                tar = np.array(tar_list)
                tar = self.downsample(tar, self.ds_target, self.inputs_dtype, self.device, self.target_transforms, is_mask=True)
                # tar = np.moveaxis(tar, 0, -1)  # make sure you fix that in the transforms
                tar = tar.astype(np.int_, copy=False)

                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x = Image.open(str(input_ID))
            tar_list = []
            for target in target_ID:
                tar_list.append(Image.open(str(target)))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

            # convert to numpy
            x = np.array(x)
            x = self.downsample(x, self.ds_input, self.inputs_dtype, self.device, self.input_transforms)


            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            y = np.array(tar_list)
            y = self.downsample(y, self.ds_target, self.inputs_dtype, self.device, self.target_transforms, is_mask=True)

            #y = np.moveaxis(y, 0, -1)  # make sure you fix that in the transforms


        if self.augmenter is not None:
            # self.augmenter.ignore_channels = random.sample(range(3), 2)
            augmented = self.augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y

    @staticmethod
    def downsample(img, ds_model, img_dtype, device, transform=None, is_mask=False):

        if transform is not None:
            img = transform(img)

        img = torch.from_numpy(img).type(img_dtype)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)

        with torch.no_grad():
            pr = ds_model(img)

        if is_mask:
            pr = (pr > 0.5).type(pr.dtype)

        pr = torch.squeeze(pr.cpu(), dim=0).numpy()

        pr = np.moveaxis(pr, 0, -1)

        return pr



class DataSetTask3CropRbSP(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""

    def __init__(
        self,
        inputs: list,
        targets: dict,
        n_classes: int,
        transform=None,
        augmenter=None,
        use_cache=False,
        pre_transform=None,
        resize=True,
        new_size=(512, 288),
    ):

        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.augmenter = augmenter
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.n_classes = n_classes
        self.resize = resize
        self.new_size = new_size


        if self.use_cache:
            self.cached_data = []
            progressbar = tqdm(range(len(self.inputs)), desc="Caching")
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                # Load input and target
                img = Image.open(str(img_name))
                tar_list = []
                for target in tar_name:
                    tar_list.append(Image.open(str(target)))

                # resizing
                if self.resize:
                    img = img.resize(self.new_size, resample=Image.LANCZOS)
                    tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

                # convert to numpy
                img = np.array(img)
                tar_list = [np.array(target) for target in tar_list]
                # if n_classes > 1:
                #     target[target == 100] = n_classes - 1
                tar = np.array(tar_list)
                tar = np.moveaxis(tar, 0, -1)

                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        if self.use_cache:
            x, y = self.cached_data[index]

        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x = Image.open(str(input_ID))
            tar_list = []
            for target in target_ID:
                tar_list.append(Image.open(str(target)))

            # resizing
            if self.resize:
                x = x.resize(self.new_size, resample=Image.LANCZOS)
                tar_list = [target.resize(self.new_size, resample=Image.LANCZOS) for target in tar_list]

            # convert to numpy
            x = np.array(x)
            tar_list = [np.array(target) for target in tar_list]
            # if n_classes > 1:
            #     target[target == 100] = n_classes - 1
            y = np.array(tar_list)
            y = np.moveaxis(y, 0, -1)

        if self.augmenter is not None:
            augmented = self.augmenter(image=x, mask=y)
            x, y = augmented['image'], augmented['mask']

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(
            self.targets_dtype
        )

        return x, y