import os
import cv2
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision.datasets.folder import pil_loader
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
import jpeg4py as jpeg
from extras import *
from image_utils import *
from utils import *
from augmentations import * #get_transforms
from preprocessing import *



class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, phase, cfg):
        """
        Args:
                fold: for k fold CV
                images_folder: the folder which contains the images
                df_path: data frame path, which contains image ids
                transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.phase = phase
        self.df = df
        self.size = cfg['size']
        self.num_samples = self.df.shape[0]
        self.fnames = self.df["id_code"].values
        self.labels = self.df["diagnosis"].values.astype("int64")
        self.num_classes = len(np.unique(self.labels))
        self.labels = to_multi_label(self.labels, self.num_classes)  # [1]
        # self.labels = np.eye(self.num_classes)[self.labels]
        self.transform = get_transforms(phase, cfg)
        self.root = os.path.join(cfg['home'], cfg['data_folder'])

        '''
        self.images = []
        for fname in tqdm(self.fnames):
            path = os.path.join(self.images_folder, "bgcc300", fname + ".npy")
            image = np.load(path)
            self.images.append(image)
        '''

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        label = self.labels[idx]
        #path = os.path.join(self.root, fname)
        path = os.path.join(self.root, fname.split('.')[0] + '.npy')
        image = np.load(path)
        #image = resize_sa(image, self.size)
        #print(image.shape)
        #filename, ext = os.path.splitext(path)
        #if ext == ".jpeg" or ext == ".jpg":
        #    image = jpeg.JPEG(path).decode()
        #else:
        #    image = Image.open(path)
        #    image = np.array(image)
        #image = cv2.imread(path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = resize_sa(image, self.size)
        #image = id_to_image(path, size=self.size, clahe=True)
        image = self.transform(image=image)["image"]
        return fname, image, label

    def __len__(self):
        #return 100
        return len(self.df)


def get_sampler(df, cfg):
    if cfg['cw_sampling']:
        '''sampler using class weights (cw)'''
        class_weights = cfg['class_weights']
        print("weights", class_weights)
        dataset_weights = [class_weights[idx] for idx in df["diagnosis"]]
    elif cfg['he_sampling']:
        '''sampler using hard examples (he)'''
        print('Hard example sampling')
        dataset_weights = df["weight"].values
    else:
        return None
    datasampler = sampler.WeightedRandomSampler(dataset_weights, len(df))
    return datasampler

def resampled(df, count_dict):
    ''' resample from df with replace=False'''
    def sample(obj):  # [5]
        return obj.sample(n=count_dict[obj.name], replace=False, random_state=69)
    sampled_df = df.groupby('diagnosis').apply(sample).reset_index(drop=True)
    return sampled_df


def provider(phase, cfg):
    HOME = cfg['home']
    if cfg['pretraining']:
        df_path = cfg['old_df_path']
    else:
        df_path = cfg['new_df_path']
    df = pd.read_csv(os.path.join(HOME, df_path))
    #print(df['diagnosis'].value_counts())
    df['weight'] = 1 # [10]

    if cfg['he_sampling']:
        hard_examples = pd.read_csv(cfg['hard_df']).index.tolist()
        df.at[hard_examples, 'weight'] = cfg['hard_ex_weight']

    if cfg['tc_dups']:
        '''remove dups before train/val split'''
        bad_dups = np.load(os.path.join(HOME, cfg["bad_idx"]))
        good_dups = np.load(os.path.join(HOME, cfg["dups_wsd"]))  # [3]
        good_dups_df = df.iloc[good_dups] # to be added later on
        all_dups = np.array(list(bad_dups) + list(good_dups))
        df = df.drop(df.index[all_dups])


    # remove class 0, subtract all by 1
    #df = df[df['diagnosis'] != 0]
    #df['diagnosis'] -= 1


    if cfg['sample']: #used in old data training
        count_dict = cfg['count_dict']
        df = resampled(df, count_dict)

    if cfg['pseudo']:
        pseudo_test = pd.read_csv(cfg['pseudo_df'])
        df.append(pseudo_test, ignore_index=True)
    #print(df['diagnosis'].value_counts())

    fold = cfg['fold']
    total_folds = cfg['total_folds']
    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(df["id_code"], df["diagnosis"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    if cfg['add_old_samples'] and phase == "train":
        sample_dict = cfg['sample_dict']
        df_old = pd.read_csv(cfg['old_df_path'])
        sampled_df_old = resampled(df_old, sample_dict)
        train_df = train_df.append(sampled_df_old, ignore_index=False)

    #'''test'''
    #extra = pd.read_csv('data/2015.csv')
    #sampled_extra = resampled(extra, cfg)
    #train_df = train_df.append(sampled_extra, ignore_index=False)
    #print(f'data dist:\n {train_df["diagnosis"].value_counts(normalize=True)}\n')
    #'''test over'''

    if cfg['messidor_in_train']:
        mes_df = pd.read_csv(cfg['mes_df'])
        #mes_df = mes_df[mes_df.diagnosis != 3] # drop class 3, see [12]
        #mes_df = mes_df.replace(to_replace=3, value=3.5)
        mes_df['weight'] = 1
        train_df = train_df.append(mes_df, ignore_index=True)

    if cfg['idrid_in_train']:
        idrid_df = pd.read_csv(cfg['idrid_df'])
        idrid_df['weight'] = 1
        train_df = train_df.append(idrid_df, ignore_index=True)
        #df = df.append(mes_df, ignore_index=True)

    #'''test'''
    ## val set dist => public test set dist
    ## line 170 modified too.

    #sample_dict = {
    #        2:    0.647303,
    #        1:    0.185166,
    #        1:    0.070539,
    #        3:    0.058091,
    #        4:    0.038900
    #}

    #count = {}
    #for i, j in sample_dict.items():
    #    count[i] = int(sample_dict[i] * 650) # so that, val size is ~20%

    #def sample(obj):  # [5]
    #    return obj.sample(n=count[obj.name], replace=False, random_state=69)

    #val_df = df.groupby('diagnosis').apply(sample).reset_index(drop=True)
    #train_df = df[~df['id_code'].isin(val_df.id_code.tolist())]

    #'''test over'''

    if 'folder' in cfg.keys():
        # save for analysis, later on
        train_df.to_csv(os.path.join(HOME, cfg['folder'], 'train.csv'), index=False)
        val_df.to_csv(os.path.join(HOME, cfg['folder'], 'val.csv'), index=False)

    if phase == "train":
        df = train_df.copy()
    elif phase == "val":
        df = val_df.copy()
    elif phase == "val_new": # [11]
        df = pd.read_csv('data/train.csv')

    print(f"{phase}: {df.shape}")
    print(f'Data dist:\n {df["diagnosis"].value_counts(normalize=True)}\n')

    df = df.sample(frac=1, random_state=69) # shuffle

    #df = pd.read_csv(cfg['diff_path'])

    image_dataset = ImageDataset(df, phase, cfg)

    datasampler = None
    if phase == "train":
        datasampler = get_sampler(df, cfg)
    print(f'datasampler: {datasampler}')

    batch_size = cfg['batch_size'][phase]
    num_workers = cfg['num_workers']
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False if datasampler else True,
        sampler=datasampler,
    )  # shuffle and sampler are mutually exclusive args

    #print(f'len(dataloader): {len(dataloader)}')
    return dataloader



def testprovider(cfg):
    HOME = cfg['home']
    df_path = cfg['sample_submission']
    df = pd.read_csv(os.path.join(HOME, df_path))
    phase = cfg['phase']
    if phase == "test":
        df['id_code'] += '.png'
    batch_size = cfg['batch_size']['test']
    num_workers = cfg['num_workers']


    dataloader = DataLoader(
        ImageDataset(df, phase, cfg),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    return dataloader


if __name__ == "__main__":
    ''' doesn't work, gotta set seeds at function level
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    '''
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=""
    import torchvision
    torchvision.set_image_backend('accimage')

    import time
    start = time.time()
    phase = "train"
    args = get_parser()
    cfg = load_cfg(args)
    cfg["num_workers"] = 8
    cfg["batch_size"]["train"] = 4
    cfg["batch_size"]["val"] = 4

    dataloader = provider(phase, cfg)
    ''' train val set sanctity
    #pdb.set_trace()
    tdf = dataloader.dataset.df
    phase = "val"
    dataloader = provider(phase, cfg)
    vdf = dataloader.dataset.df
    print(len([x for x in tdf.id_code.tolist() if x in vdf.id_code.tolist()]))
    exit()
    '''
    total_labels = []
    total_len = len(dataloader)
    from collections import defaultdict
    fnames_dict = defaultdict(int)
    for idx, batch in enumerate(dataloader):
        fnames, images, labels = batch
        labels = (torch.sum(labels, 1) - 1).numpy().astype('uint8')
        for fname in fnames:
            fnames_dict[fname] += 1

        print("%d/%d" % (idx, total_len), images.shape, labels.shape)
        total_labels.extend(labels.tolist())
        #pdb.set_trace()
    print(np.unique(total_labels, return_counts=True))
    diff = time.time() - start
    print('Time taken: %02d:%02d' % (diff//60, diff % 60))

    print(np.unique(list(fnames_dict.values()), return_counts=True))
    #pdb.set_trace()


"""
Footnotes:

https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition

[1] CrossEntropyLoss doesn't expect inputs to be one-hot, but indices
[2] .value_counts() returns in descending order of counts (not sorted by class numbers :)
[3]: bad_indices are those which have conflicting diagnosises, duplicates are those which have same duplicates, we shouldn't let them split in train and val set, gotta maintain the sanctity of val set
[4]: used when the dataframe include external data and we want to sample limited number of those
[5]: as replace=False,  total samples can be a finite number so that those many number of classes exist in the dataset, and as the count_dist is approx, not normalized to 1, 7800 is optimum, totaling to ~8100 samples

[6]: albumentations.Normalize will divide by 255, subtract mean and divide by std. output dtype = float32. ToTensor converts to torch tensor and divides by 255 if input dtype is uint8.
[7]: indices of hard examples, evaluated using 0.81 scoring model.
[10]: messidor df append will throw err when doing hard ex sampling.
[11]: using current comp's data as val set in old data training.
[12]: messidor's class 3 is class 3 and class 4 combined.
"""
