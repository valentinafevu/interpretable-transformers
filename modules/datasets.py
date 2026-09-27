import os

import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms


class Dataset:
    def __init__(self):
        self.classes = ()
        self.n_channels = None
        self.n_classes = 0
        self.name = ""
        self.trainset = None
        self.testset = None
        self.train_transform = None
        self.test_transform = None
        self.multi_label = False

    def get_samples_from_indices(self, indices, device="mps", set="test"):
        dataset = self.testset if set == "test" else self.trainset
        if dataset is None:
            raise ValueError("Dataset not loaded. Call get_loaders() first.")
        images = torch.stack([dataset[i][0] for i in indices]).to(device)
        label_list = [dataset[i][1] for i in indices]
        if torch.is_tensor(label_list[0]):
            labels = torch.stack(label_list).to(device)
        else:
            labels = torch.tensor(label_list).to(device)
        return images, labels

    def visualize_sample(self, indices=None, return_images=False, set="test"):
        """
        Visualizes a batch of random images if the indices are not provided, otherwise visualizes the images corresponding to the provided indices.
        Creates a grid of images with their corresponding labels as titles.
        """
        if (self.testset is None and set == "test") or (self.trainset is None and set == "train"):
            raise ValueError("Dataset not loaded. Call get_loaders() first.")

        if indices is None:
            dataset = self.testset if set == "test" else self.trainset
            indices = torch.randperm(len(dataset))[:16].tolist()

        images, labels = self.get_samples_from_indices(indices, device="cpu", set=set)
        labels = labels.tolist()
        images = images * 0.5 + 0.5  # unnormalize

        n = len(indices)
        ncols = min(8, n)
        nrows = (n + ncols - 1) // ncols

        _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = [axes] if n == 1 else axes.flatten() if nrows > 1 else list(axes)

        for ax, img, label in zip(axes, images, labels):
            ax.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray' if self.n_channels == 1 else None)
            if self.multi_label:
                title = ", ".join(c for c, v in zip(self.classes, label) if v)
                ax.set_title(title, fontsize=6)
            else:
                ax.set_title(self.classes[label])
            ax.axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()
        if return_images:
            return images, labels


class CIFAR10Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.n_channels = 3
        self.n_classes = len(self.classes)
        self.name = "CIFAR10"

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.333), interpolation=2),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def get_loaders(self, batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.train_transform)
        self.trainset = trainset
        if train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:train_sample_size]
            trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=self.test_transform)
        self.testset = testset
        if test_sample_size is not None:
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return trainloader, testloader, self.classes


class MNISTDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.classes = tuple(str(i) for i in range(10))
        self.n_channels = 1
        self.n_classes = len(self.classes)
        self.name = "MNIST"

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(padding=4, fill=0),
            transforms.RandomRotation(degrees=(-90, 90), fill=(0,)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),
            transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.0), ratio=(0.75, 1.333), interpolation=2),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def get_loaders(self, batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=self.train_transform)
        self.trainset = trainset
        if train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:train_sample_size]
            trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=self.test_transform)
        self.testset = testset
        if test_sample_size is not None:
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return trainloader, testloader, self.classes


class _CelebACSVDataset(torch.utils.data.Dataset):
    """
    Reads the Kaggle export of CelebA (jessicali9530/celeba-dataset) instead of
    torchvision.datasets.CelebA, whose annotation files are only served from
    Google Drive and routinely hit its download quota.

    Expects `root` to directly contain list_attr_celeba.csv, list_eval_partition.csv
    and an img_align_celeba/ folder (optionally nested one level deeper, as Kaggle
    zips it: img_align_celeba/img_align_celeba/*.jpg).
    """

    _SPLIT_TO_PARTITION = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, root, split='train', transform=None):
        self.transform = transform

        img_dir = os.path.join(root, 'img_align_celeba')
        if os.path.isdir(os.path.join(img_dir, 'img_align_celeba')):
            img_dir = os.path.join(img_dir, 'img_align_celeba')
        self.img_dir = img_dir

        attr_df = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'), index_col='image_id')
        self.attr_names = tuple(attr_df.columns)
        attr_df = (attr_df + 1) // 2  # map {-1, 1} -> {0, 1}

        if split != 'all':
            partition_df = pd.read_csv(os.path.join(root, 'list_eval_partition.csv'), index_col='image_id')
            partition = self._SPLIT_TO_PARTITION[split]
            keep_ids = partition_df.index[partition_df['partition'] == partition]
            attr_df = attr_df.loc[attr_df.index.intersection(keep_ids)]

        self.image_ids = attr_df.index.tolist()
        self.attr = torch.tensor(attr_df.values, dtype=torch.long)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.img_dir, self.image_ids[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.attr[index]



class CelebADataset(Dataset):
    """
    Multi-label attribute classification: each sample's label is a 40-dim
    binary vector (one entry per facial attribute), not a single class index.
    Loads from a Kaggle-style CelebA export (see _CelebACSVDataset) rather than
    torchvision.datasets.CelebA, since the latter's annotation files are only
    served from Google Drive and routinely hit its download quota.
    """

    def __init__(self, image_size=128, root='./data/celeba'):
        super().__init__()
        self.classes = (
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
            'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
            'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
            'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
            'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young',
        )
        self.n_channels = 3
        self.n_classes = len(self.classes)
        self.name = "CelebA"
        self.image_size = image_size
        self.multi_label = True
        self.root = root

        self.train_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def get_loaders(self, batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        trainset = _CelebACSVDataset(root=self.root, split='train', transform=self.train_transform)
        self.trainset = trainset
        if train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:train_sample_size]
            trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = _CelebACSVDataset(root=self.root, split='test', transform=self.test_transform)
        self.testset = testset
        if test_sample_size is not None:
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return trainloader, testloader, self.classes


class OxfordIIITPetDataset(Dataset):
    def __init__(self, image_size=128):
        super().__init__()
        self.classes = (
            'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound',
            'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair',
            'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter',
            'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond',
            'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian',
            'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed',
            'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier',
            'Wheaten Terrier', 'Yorkshire Terrier',
        )
        self.n_channels = 3
        self.n_classes = len(self.classes)
        self.name = "OxfordIIITPet"
        self.image_size = image_size

        to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))

        self.train_transform = transforms.Compose([
            to_rgb,
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.333), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            to_rgb,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def get_loaders(self, batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
        trainset = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval',
                                                       target_types='category', download=True,
                                                       transform=self.train_transform)
        self.trainset = trainset
        if train_sample_size is not None:
            indices = torch.randperm(len(trainset))[:train_sample_size]
            trainset = torch.utils.data.Subset(trainset, indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test',
                                                      target_types='category', download=True,
                                                      transform=self.test_transform)
        self.testset = testset
        if test_sample_size is not None:
            indices = torch.randperm(len(testset))[:test_sample_size]
            testset = torch.utils.data.Subset(testset, indices)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        return trainloader, testloader, self.classes



# ----------------------------------------------------------------------------
# Animals with Attributes 2 (AwA2)
# ----------------------------------------------------------------------------

AWA2_HF_REPO = "quanda-bench-test/awa2"
AWA2_BASE_URL = "https://cvml.ista.ac.at/AwA2/AwA2-base.zip"


def _awa2_read_base_files(root):
    """
    Lee classes.txt, predicates.txt y predicate-matrix-{continuous,binary}.txt de
    AwA2-base.zip (descargándolo si hace falta). El orden de las clases es el de
    classes.txt, que es el mismo orden de filas de las matrices de predicados.
    """
    import io
    import urllib.request
    import zipfile

    import numpy as np

    base_dir = os.path.join(root, 'Animals_with_Attributes2')
    if not os.path.isfile(os.path.join(base_dir, 'predicate-matrix-continuous.txt')):
        os.makedirs(root, exist_ok=True)
        with urllib.request.urlopen(AWA2_BASE_URL) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(root)

    def read_index_file(name):
        # Formato: "<idx>\t<nombre>" por línea (idx 1-indexed)
        with open(os.path.join(base_dir, name)) as f:
            return tuple(line.split()[1] for line in f if line.strip())

    classes = read_index_file('classes.txt')
    predicates = read_index_file('predicates.txt')
    continuous = np.loadtxt(os.path.join(base_dir, 'predicate-matrix-continuous.txt'), dtype=np.float32)
    binary = np.loadtxt(os.path.join(base_dir, 'predicate-matrix-binary.txt'), dtype=np.int64)
    assert continuous.shape == binary.shape == (len(classes), len(predicates)) == (50, 85)
    return classes, predicates, continuous, binary


def _awa2_iter_hf(classes):
    """(PIL, label) desde el espejo Parquet de HF por streaming (sin guardar los ~14 GB)."""
    from datasets import load_dataset

    class_to_idx = {c: i for i, c in enumerate(classes)}
    stream = load_dataset(AWA2_HF_REPO, split='train', streaming=True)
    # El espejo HF usa orden alfabético; se remapea al orden de classes.txt por nombre.
    hf_to_idx = [class_to_idx[name] for name in stream.features['label'].names]
    for example in stream:
        yield example['image'], hf_to_idx[example['label']]


def _awa2_iter_zip(classes, zip_path):
    """(PIL, label) leyendo directo de AwA2-data.zip (sin descomprimir a disco)."""
    import zipfile

    class_to_idx = {c: i for i, c in enumerate(classes)}
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            parts = name.split('/')  # Animals_with_Attributes2/JPEGImages/<clase>/<img>.jpg
            if len(parts) == 4 and parts[1] == 'JPEGImages' and parts[3].endswith('.jpg'):
                with z.open(name) as f:
                    yield Image.open(f).convert('RGB'), class_to_idx[parts[2]]


def prepare_awa2_cache(root='./data/awa2', cache_size=64, test_fraction=0.2, seed=0,
                       source='hf', zip_path=None):
    """
    Construye un cache compacto de AwA2 en `root/awa2_{cache_size}.npz`:
      - images: uint8 (N, cache_size, cache_size, 3) — lado corto a cache_size + center crop
      - labels: int64 (N,) en el orden de classes.txt
      - split:  uint8 (N,) 0 = train, 1 = test (estratificado por clase, seed fija)

    source='hf'  -> streaming desde el espejo Parquet `quanda-bench-test/awa2`.
                    (`randall-lab/awa2` es solo un loading script que `datasets>=4` ya no
                    ejecuta, y que de todos modos baja el zip de 13.9 GB de ISTA.)
    source='zip' -> AwA2-data.zip de https://cvml.ista.ac.at/AwA2/ ya descargado en `zip_path`
                    (bastante más rápido que HF en la práctica: ~9 MB/s vs ~1 MB/s).
    Ambas fuentes son la misma colección de 37,322 imágenes.
    """
    import numpy as np
    from tqdm.auto import tqdm

    cache_path = os.path.join(root, f'awa2_{cache_size}.npz')
    if os.path.isfile(cache_path):
        return cache_path

    classes, _, _, _ = _awa2_read_base_files(root)
    if source == 'hf':
        examples = _awa2_iter_hf(classes)
    elif source == 'zip':
        examples = _awa2_iter_zip(classes, zip_path)
    else:
        raise ValueError(f"source debe ser 'hf' o 'zip', no {source!r}")

    resize = transforms.Compose([transforms.Resize(cache_size), transforms.CenterCrop(cache_size)])
    images, labels = [], []
    for image, label in tqdm(examples, desc=f'AwA2 ({source})', unit='img', total=37322):
        images.append(np.asarray(resize(image.convert('RGB')), dtype=np.uint8))
        labels.append(label)
    images = np.stack(images)
    labels = np.asarray(labels, dtype=np.int64)
    assert len(labels) == 37322 and len(np.unique(labels)) == len(classes)

    # Split estratificado: test_fraction de cada clase va a test.
    rng = np.random.default_rng(seed)
    split = np.zeros(len(labels), dtype=np.uint8)
    for c in range(len(classes)):
        idx = rng.permutation(np.flatnonzero(labels == c))
        split[idx[:int(round(test_fraction * len(idx)))]] = 1

    os.makedirs(root, exist_ok=True)
    np.savez(cache_path, images=images, labels=labels, split=split)
    return cache_path


class _AwA2CachedDataset(torch.utils.data.Dataset):
    """Devuelve (imagen_tensor, label_clase, vector_atributos_85) desde el cache .npz."""

    def __init__(self, images, labels, attributes, transform=None):
        self.images = images            # uint8 (N, S, S, 3)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.attributes = attributes    # float32 (50, 85), una fila por clase
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label, self.attributes[label]


class AwA2Dataset(Dataset):
    """
    Animals with Attributes 2: 50 clases de animales, 37,322 imágenes, y una matriz
    clase x atributo (50 x 85). Cada muestra es (imagen, label_clase, atributos_85):
    el vector de atributos es el de la clase (todas las imágenes de una clase lo comparten).

    attribute_type='continuous' -> predicate-matrix-continuous.txt normalizada de [0, 100] a [0, 1]
    attribute_type='binary'     -> predicate-matrix-binary.txt ({0, 1})
    """

    def __init__(self, image_size=32, root='./data/awa2', cache_size=64, attribute_type='continuous',
                 source='hf', zip_path=None):
        super().__init__()
        self.root = root
        self.source = source      # ver prepare_awa2_cache: 'hf' (streaming) o 'zip' (AwA2-data.zip de ISTA)
        self.zip_path = zip_path
        self.image_size = image_size
        self.cache_size = cache_size
        self.attribute_type = attribute_type
        self.n_channels = 3
        self.name = "AwA2"

        classes, predicates, continuous, binary = _awa2_read_base_files(root)
        self.classes = tuple(c.replace('+', ' ') for c in classes)
        self.n_classes = len(self.classes)
        self.attribute_names = predicates
        self.attributes_continuous = torch.tensor(continuous / 100.0, dtype=torch.float32)
        self.attributes_binary = torch.tensor(binary, dtype=torch.float32)
        self.attributes = self.attributes_continuous if attribute_type == 'continuous' else self.attributes_binary

        # Mismo esquema de augmentation que CIFAR10Dataset, sobre el cache a cache_size.
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.333), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def get_loaders(self, batch_size=4, num_workers=0, train_sample_size=None, test_sample_size=None):
        import numpy as np

        cache = np.load(prepare_awa2_cache(self.root, cache_size=self.cache_size,
                                          source=self.source, zip_path=self.zip_path))
        images, labels, split = cache['images'], cache['labels'], cache['split']

        train_mask, test_mask = split == 0, split == 1
        trainset = _AwA2CachedDataset(images[train_mask], labels[train_mask], self.attributes, self.train_transform)
        testset = _AwA2CachedDataset(images[test_mask], labels[test_mask], self.attributes, self.test_transform)
        self.trainset, self.testset = trainset, testset

        if train_sample_size is not None:
            trainset = torch.utils.data.Subset(trainset, torch.randperm(len(trainset))[:train_sample_size])
        if test_sample_size is not None:
            testset = torch.utils.data.Subset(testset, torch.randperm(len(testset))[:test_sample_size])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        return trainloader, testloader, self.classes
