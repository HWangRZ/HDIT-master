import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import os
import cv2
import torch
import random

 
# ---------------------------------------------------------------------------
#  Dataset root
#  All HSI datasets are expected under this directory. Override it with the
#  HSI_DATASET_ROOT environment variable, e.g.
#      export HSI_DATASET_ROOT=/path/to/datasets
#  Expected layout (file names must match):
#      $HSI_DATASET_ROOT/PU/PaviaU.mat
#      $HSI_DATASET_ROOT/PU/PaviaU_gt.mat
#      $HSI_DATASET_ROOT/Houston/Houston.mat
#      $HSI_DATASET_ROOT/LongKou/WHU_Hi_LongKou.mat
#      $HSI_DATASET_ROOT/Salinas/Salinas_corrected.mat
#      ...
# ---------------------------------------------------------------------------
DATASET_ROOT = os.environ.get("HSI_DATASET_ROOT", "./datasets")


def _ds(*parts):
    """Join parts onto DATASET_ROOT."""
    return os.path.join(DATASET_ROOT, *parts)


dataset_path = {

    'PU': {'corrected': _ds('PU', 'PaviaU.mat'),
           'gt': _ds('PU', 'PaviaU_gt.mat')},

    'Houston': {'corrected': _ds('Houston', 'Houston.mat'),
                'gt': _ds('Houston', 'Houston_gt.mat')},

    'LongKou': {'corrected': _ds('LongKou', 'WHU_Hi_LongKou.mat'),
                'gt': _ds('LongKou', 'WHU_Hi_LongKou_gt.mat')},

    'Salinas': {'raw': _ds('Salinas', 'Salinas.mat'),
                'corrected': _ds('Salinas', 'Salinas_corrected.mat'),
                'gt': _ds('Salinas', 'Salinas_gt.mat')},

    'IP': {'corrected': _ds('IP', 'Indian_pines_corrected.mat'),
           'gt': _ds('IP', 'Indian_pines_gt.mat')},

    'KSC': {'corrected': _ds('KSC', 'KSC.mat'),
            'gt': _ds('KSC', 'KSC_gt.mat')},

    'Botswana': {'corrected': _ds('Botswana', 'Botswana.mat'),
                 'gt': _ds('Botswana', 'Botswana_gt.mat')},

    'ZY_HHK': {'corrected': _ds('ZY_HHK', 'ZY_hhk.mat'),
               'gt': _ds('ZY_HHK', 'ZY_hhk_gt.mat')},

    'HongHu': {'corrected': _ds('HongHu', 'WHU_Hi_HongHu.mat'),
               'gt': _ds('HongHu', 'WHU_Hi_HongHu_gt.mat')},

    'GF_HHK': {'corrected': _ds('GF_HHK', 'huanghekou_data.mat'),
               'gt': _ds('GF_HHK', 'huanghekou_gt.mat')},

}

dataset_name_dict = {
    'PU': 'PaviaU',
    'PC': 'Pavia',
    'IP': 'Indian_pines',
    'Salinas': 'Salinas',
    'KSC': 'KSC',
    'Botswana': 'Botswana',
    'HHK': 'HuangHeKou',
    'LongKou': 'LongKou',
    'HongHu': 'HongHu',
    'Houston': 'Houston_University'#######not2018
}

dataset_size_dict = {
    'PU': [610, 340, 103],
    'IP': [145, 145, 200],
    'Salinas': [512, 217, 204],
    'KSC': [512, 614, 176],
    'Botswana': [1476, 256, 145],
    'ZY_HHK': [1147, 1600, 119],
    'Houston': [349, 1905, 144],
    'LongKou': [550, 400, 270],
    'HongHu': [940, 475, 270],
    'GF_HHK': [1342, 1185, 285],
}

dataset_class_dict = {

    'PU': ["Asphalt", "Meadows", "Gravel", "Trees",
           "Painted metal sheets", "Bare Soil", "Bitumen",
           "Self-Blocking Bricks", "Shadows"],

    'IP': ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
           "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
           "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"],

    'Salinas': ["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow",
                "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
                "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"],

    'KSC': ["Scrub", "Willow swamp", "Cabbage palm hammock", "Cabbage palm/oak hammock",
            "Slash pine", "Oak/broadleaf hammock", "Hardwood swamp", "Graminoid marsh",
            "Spartina marsh", "Cattail marsh", "Salt marsh", "Mud flats", "Wate"],

    'Botswana': ["Water", "Hippo grass", "Floodplain grasses 1", "Floodplain grasses 2", "Reeds", "Riparian",
                 "Firescar", "Island interior", "Acacia woodlands", "Acacia shrublands", "Acacia grasslands",
                 "Short mopane", "Mixed mopane", "Exposed soils"],

    'ZY_HHK': ["Reed", "Spartina alterniflora", "Salt filter pond", "Salt evaporation pond", "Dry pond", "Tamarisk",
               "Salt pan", "Seepweed", "River", "Sea", "Mudbank", "Tidal creek", "Fallow land",
               "Ecological restoration pond", "Robinia", "Fishpond", "Pit pond", "Building", "Bare land", "Paddyfield",
               "Cotton", "Soybean", "Corn"],

    'Houston': ["Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water", "Residential",
                "Commercial", "Road", "Highway", "Railway", "Parking Lot1", "Parking Lot2", "Tennis court",
                "Running track"],

    'LongKou': ["Corn", "Cotton", "Sesame", "Broad-leaf soybean", "Narrow-leaf soybean",
                "Rice", "Water", "Roads and houses", "Mixed weed"],

    
    'HongHu': ["Red roof", "Road", "Bare soil", "Cotton", "Cotton firewood", "Rape", "Chinese cabbage", "Pakchoi", "Cabbage", "Tuber mustard", "Brassica parachinensis", "Brassica chinensis", "Small Brassica chinensis", "Lactuca sativa", "Celtuce", "Film covered lettuce", "Romaine lettuce", "Carrot", "White radish", "Garlic sprout", "Broad bean", "Tree"],

    'GF_HHK': ["Aquaculture", "Seep sea", "Soybean", "Rice", "Building", "Maize", "Broomcorn",
               "Locust", "Spartina", "Shallow sea", "Mud flat", "River", "Suaeda salsa", "Reed",
               "Salt marsh", "Intertidal saltwater", "Tamarix", "Pond", "Flood plain",
               "Freshwater herbaceous marsh", "Aquatic vegetation"]

}

color_map_dict = {

    'PU': np.array([[0, 0, 255], [76, 230, 0], [255, 190, 232], [255, 0, 0], [156, 156, 156],
                    [255, 255, 115], [0, 255, 197], [132, 0, 168], [0, 0, 0]], dtype=np.uint8),

    'IP': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                    [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                    [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                    [255, 170, 0]], dtype=np.uint8),

    'Salinas': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                         [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                         [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0],
                         [255, 170, 0]], dtype=np.uint8),

    'KSC': np.array([[0, 168, 132], [76, 0, 115], [255, 0, 0], [190, 255, 232], [0, 0, 0],
                     [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                     [0, 0, 255], [156, 156, 156], [115, 223, 255]], dtype=np.uint8),

    'Botswana': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0],
                          [115, 0, 0], [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223],
                          [0, 0, 255], [156, 156, 156], [115, 223, 255], [0, 255, 0]], dtype=np.uint8),

    'Houston': np.array([[0, 168, 132], [76, 0, 115], [0, 0, 0], [190, 255, 232], [255, 0, 0], [115, 0, 0],
                         [205, 205, 102], [137, 90, 68], [215, 158, 158], [255, 115, 223], [0, 0, 255],
                         [156, 156, 156], [115, 223, 255], [0, 255, 0], [255, 255, 0]], dtype=np.uint8),

    'LongKou': np.array([[255, 0, 0], [240, 155, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 138, 138],
                         [0, 0, 255], [0, 0, 0], [160, 32, 240]], dtype=np.uint8),

    'GF_HHK': np.array([[128, 255, 0], [0, 30, 190], [218, 112, 213], [0, 138, 140], [255, 128, 80], [255, 255, 0],
                        [47, 139, 88], [0, 255, 0], [255, 165, 0], [128, 255, 212], [204, 0, 0], [140, 0, 0],
                        [0, 0, 140], [254, 0, 0], [218, 112, 213], [65, 105, 226], [0, 140, 0], [255, 0, 255],
                        [245, 164, 98], [0, 255, 255], [0, 0, 254]], dtype=np.uint8),

}

false_color_dict = {
    'PU': [102, 56, 31],
    'IP': [50, 27, 17],
    'Salinas': [57, 27, 17],
    'KSC': [],
    'Botswana': [],
    'Houston': [80, 59, 40],
    'LongKou': [180, 126, 72],
}

true_color_dict = {
    'PU': [56, 31, 6],
    'IP': [27, 17, 7],
    'Salinas': [27, 17, 7],
    'KSC': [],
    'Botswana': [],
}


def load_dataset(dataset_name: str, key: int):#############
    """
    load data
    :param dataset_name: dataset's dictionary
    :param key: indicator
    :return: numpy.ndarray
    """
    kv = {0: 'raw',
          1: 'corrected',
          2: 'gt'}
    path = dataset_path[dataset_name]
    try:
        data = sio.loadmat(path[kv[key]])
        for item in data.items():
            if type(item[1]) is np.ndarray:
                return item[1]
    
    except:
        import h5py
        data = h5py.File(path[kv[key]], 'r')
        data = data[path[kv[key]].split('/')[-1].split('.')[0]][:]
        if dataset_name == 'GF_HHK' and key == 1:
            return np.transpose(data, (1, 2, 0))
        else:
            return data


def pca_processing(data, n_pc: int, whiten=True):########
    """
    applying pca
    :param data: 
    :param n_components: 
    :param whiten: 
    :return: data after applying pca
    """
    h, w, c = data.shape
    data = np.reshape(data, (-1, c))
    pca = PCA(n_components=n_pc, whiten=whiten)
    data = pca.fit_transform(data)
    return np.reshape(data, (h, w, n_pc))


default_mirror_width = 35
def mirror_concatenate(x, mirror_width=default_mirror_width):
    return cv2.copyMakeBorder(x, mirror_width, mirror_width, mirror_width, mirror_width, cv2.BORDER_REFLECT)


def HSI_LazyProcessing(dataset_name='PU', n_pc=16, no_processing=False, whiten=True):
    """

    :param dataset_name:
    :param n_pc:
    :param patch_size:
    :return:
    """
    # patch_radius = patch_size // 2
    Y = load_dataset(dataset_name, key=2)
    assert n_pc > 0
    if whiten:
        pca_file_path = './save/pca_result/' + dataset_name + '_mirror_pca_whiten.npy'
    else:
        pca_file_path = './save/pca_result/' + dataset_name + '_mirror_pca.npy'

    if no_processing:
        X_extension = load_dataset(dataset_name, key=1)
        row, col, band = X_extension.shape
        X_extension = X_extension.astype(np.float32)
        X_extension = (X_extension - np.mean(X_extension, axis=0)) / np.std(X_extension, axis=0)
        Y = Y.reshape(row * col, -1)
        return X_extension, Y, [row, col, band]

    if os.path.exists(pca_file_path) and n_pc != 0:
        X_extension = np.load(pca_file_path)
        X_extension = X_extension[..., :n_pc]
    else:
        if not os.path.exists('./save/pca_result'):
            os.makedirs('./save/pca_result')
        X = load_dataset(dataset_name, key=1)
        [row, col, band] = X.shape
        X = pca_processing(X, n_pc=band, whiten=whiten)
        np.save(pca_file_path, X)
        X_extension = X[..., :n_pc]

    row, col = Y.shape
    band = X_extension.shape[2]
    Y = Y.reshape(row * col, -1)

    return X_extension, Y, [row, col, band]




def data_augmentation(patch):   #数据增强
    '''

    :param p: the probability of execute data augmentation
    :return:
    '''

    def vertical_rotation(patch):

        return np.flip(patch, axis=0)

    def horizontal_rotation(patch):

        return np.flip(patch, axis=1)

    def transpose(patch):

        return patch.transpose((1, 0, 2))

    patch = vertical_rotation(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch
    patch = horizontal_rotation(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch
    patch = transpose(patch) if np.random.choice([0, 1], p=[1/2, 1/2]) else patch
    return patch

def split_train_test_set(Y, dataset_name='PU', train_num=5, batch_size=64, seed=0):
    """

    :param Y:
    :param dataset_name:
    :param train_num:
    :param batch_size:
    :param seed:
    :param resample_seed:
    :param growth_area:
    :return:
    """
    if isinstance(train_num, int):
        train_num = [train_num] * len(dataset_class_dict.get(dataset_name))
    elif isinstance(train_num, list):
        pass

    n_class = Y.max()
    train_set, test_set = [], []

    random_state = np.random.RandomState(seed=seed)
    for i in range(1, n_class + 1):
        index = np.where(Y == i)[0]
        #  variable n_data is used only when data is split by percentage rule.
        n_data = index.shape[0]
        random_state.shuffle(index)

        print('Preparing training samples -- ' + str(i) + ' - ' + str(n_class))
        if train_num[i - 1] * 2 > n_data:
            train_set.extend(index[:int(np.ceil(n_data / 2))])
            test_set.extend(index[int(np.ceil(n_data / 2)):])
        else:
            train_set.extend(index[:train_num[i - 1]])
            test_set.extend(index[train_num[i - 1]:])

    if len(train_set) < batch_size:
        train_batch_size = len(train_set)
    else:
        train_batch_size = batch_size

    train_set, test_set = np.array(train_set), np.array(test_set)
    train_step = int(np.ceil(train_set.size / train_batch_size))
    test_step = int(np.ceil(test_set.size / batch_size))

    # train_set and test_set represent the index of train samples and test samples
    return train_set, train_step, test_set, test_step

def generate_batch(train_test_set, X_PCAMirror, Y, dataset_name='PU', patch_size=9, batch_size=64,
                   shuffle=True, mode='train', augment=False):
    """

    :param train_test_set:
    :param X_PCAMirror:
    :param Y:
    :param label:
    :param dataset_name:
    :param patch_size:
    :param batch_size:
    :param shuffle:
    :param mode:
    :param process:
    :param augment:
    :param resample_seed:
    :param growth_area:
    :return:
    """

    num_samples = train_test_set.size
    row, col, band = dataset_size_dict.get(dataset_name)
    patch_radius = patch_size // 2

    set_idx = np.arange(train_test_set.size)
    if shuffle:
        random_state = np.random.RandomState()
        random_state.shuffle(set_idx)
    train_test_set = train_test_set[set_idx]


    for i in range(0, num_samples, batch_size):
        # batch_i represents the i-th element in current batch
        batch_i = train_test_set[np.arange(i, min(num_samples, i + batch_size))]
        batch_i_row = np.floor(batch_i * 1.0 / col).astype(np.int32)
        batch_i_col = (batch_i - batch_i_row * col).astype(np.int32)
        upper_edge, bottom_edge = (batch_i_row - patch_radius), (batch_i_row + patch_radius + 1)
        left_edge, right_edge = (batch_i_col - patch_radius), (batch_i_col + patch_radius + 1)

        patches = []
        for j in range(batch_i.size):
            patch = X_PCAMirror[
                    upper_edge[j] + patch_radius: bottom_edge[j] + patch_radius,
                    left_edge[j] + patch_radius: right_edge[j] + patch_radius,
                    :]
            patch = data_augmentation(patch) if mode == 'train' and augment else patch
            patches.append(patch)
        patches = np.array(patches)
        patches = np.transpose(patches, (0, 3, 1, 2))
        labels = Y[batch_i, :] - 1
        yield patches, labels



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_patches_from_processed_data0(X, Y, shape_info, patch_size=32, remove_zeros=True):
    """
    从已处理的HSI数据直接创建patches
    """
    row, col, band = shape_info
    
    # 重塑标签为2D格式 (row, col)
    Y_2d = Y.reshape(row, col)
    
    # 零填充函数
    def pad_with_zeros(X, margin):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
        return newX
    
    # 提取patches
    margin = patch_size // 2
    padded_X = pad_with_zeros(X, margin)
    
    patches_data = []
    patches_labels = []
    
    # 遍历每个像素位置提取patch
    for r in range(margin, padded_X.shape[0] - margin):
        for c in range(margin, padded_X.shape[1] - margin):
            patch = padded_X[r - margin:r + margin, c - margin:c + margin]
            patches_data.append(patch)
            patches_labels.append(Y_2d[r - margin, c - margin])
    
    patches_data = np.array(patches_data)
    patches_labels = np.array(patches_labels)
    
    # 移除背景标签（如果要求）
    if remove_zeros:
        mask = patches_labels > 0
        patches_data = patches_data[mask]
        patches_labels = patches_labels[mask] - 1  # 标签从0开始
    
    return patches_data, patches_labels

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    

from tqdm import tqdm
import numpy as np

def create_patches_from_processed_data1(X, Y, shape_info, patch_size=32, remove_zeros=True):

    """
    从已处理的HSI数据直接创建patches - 带精准进度条版本
    """
    row, col, band = shape_info
    
    # 重塑标签为2D格式 (row, col)
    Y_2d = Y.reshape(row, col)
    
    # 零填充函数
    def pad_with_zeros(X, margin):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
        newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
        return newX
    
    # 提取patches
    margin = patch_size // 2
    padded_X = pad_with_zeros(X, margin)
    
    patches_data = []
    patches_labels = []
    
    # 计算总工作量
    total_pixels = (padded_X.shape[0] - 2 * margin) * (padded_X.shape[1] - 2 * margin)
    print(f"🔄 开始创建patches...")
    print(f"📊 图像尺寸: {row} × {col} × {band}")
    print(f"🎯 Patch尺寸: {patch_size} × {patch_size}")
    print(f"📈 预计处理: {total_pixels:,} 个像素位置")
    
    # 创建进度条
    with tqdm(total=total_pixels, desc="创建Patches", 
              bar_format="{l_bar}{bar:40}| {percentage:3.0f}% | {n_fmt}/{total_fmt} | 耗时: {elapsed} | 剩余: {remaining} | 速度: {rate_fmt}") as pbar:
        
        # 遍历每个像素位置提取patch
        for r in range(margin, padded_X.shape[0] - margin):
            for c in range(margin, padded_X.shape[1] - margin):
                patch = padded_X[r - margin:r + margin, c - margin:c + margin]
                patches_data.append(patch)
                patches_labels.append(Y_2d[r - margin, c - margin])
                
                # 更新进度条
                pbar.update(1)
    
    patches_data = np.array(patches_data)
    patches_labels = np.array(patches_labels)
    
    # 移除背景标签（如果要求）
    if remove_zeros:
        mask = patches_labels > 0
        original_count = len(patches_data)
        patches_data = patches_data[mask]
        patches_labels = patches_labels[mask] - 1  # 标签从0开始
        final_count = len(patches_data)
        print(f"✅ 背景过滤: {original_count:,} → {final_count:,} (移除 {original_count - final_count:,} 个背景patch)")
    
    print(f"🎉 最终创建了 {len(patches_data):,} 个有效patches")
    return patches_data, patches_labels



import numpy as np

def create_patches_from_processed_data(X, Y, shape_info, patch_size=32, label_condition="!=0"):
    """
    从已处理的HSI数据直接创建patches - 优化版本（向量化+进度条）
    参数:
        X: 预处理后的高光谱数据，形状 (row, col, band)
        Y: 标签，形状 (row*col,)
        shape_info: (row, col, band)
        patch_size: 每个patch的大小
        label_condition: 控制提取patch的条件，可选：
                         - "!=0": 只提取非0标签（默认）
                         - "==0": 只提取0标签
                         - "all": 提取全部像素
                         - int，如 1: 只提取标签==1的像素
                         - list，如 [1,2,3]: 只提取标签属于这些值的像素
    返回:
        patches_data: 提取的patch数组 (N, patch_size, patch_size, band)
        patches_labels: 对应标签 (N,)
    """
    from tqdm.auto import tqdm
    
    row, col, band = shape_info
    Y_2d = Y.reshape(row, col)
    
    margin = patch_size // 2
    
    print(f"\n🔄 开始创建patches...")
    print(f"📊 图像尺寸: {row} × {col} × {band}")
    print(f"🎯 Patch尺寸: {patch_size} × {patch_size}")
    
    padded_X = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    
    if label_condition == "all":
        valid_positions = np.arange(row * col)
    elif label_condition == "!=0":
        valid_positions = np.where(Y.flatten() != 0)[0]
    elif label_condition == "==0":
        valid_positions = np.where(Y.flatten() == 0)[0]
    elif isinstance(label_condition, int):
        valid_positions = np.where(Y.flatten() == label_condition)[0]
    elif isinstance(label_condition, (list, tuple, set)):
        valid_positions = np.where(np.isin(Y.flatten(), list(label_condition)))[0]
    else:
        raise TypeError("label_condition 必须是 str、int 或 list/tuple/set")
    
    total_patches = len(valid_positions)
    print(f"📈 将创建 {total_patches:,} 个patches")
    
    if total_patches == 0:
        print("⚠️  警告: 没有找到符合条件的像素!")
        return np.array([]), np.array([])
    
    patches_data = np.zeros((total_patches, patch_size, patch_size, band), dtype=X.dtype)
    patches_labels = np.zeros(total_patches, dtype=Y.dtype)
    
    for idx, pos in enumerate(tqdm(valid_positions, desc="创建Patches", 
                                    bar_format="{l_bar}{bar:40}| {percentage:3.0f}% | {n_fmt}/{total_fmt} | 耗时: {elapsed} | 剩余: {remaining} | 速度: {rate_fmt}")):
        r = pos // col
        c = pos % col
        
        r_start = r
        r_end = r + patch_size
        c_start = c
        c_end = c + patch_size
        
        patches_data[idx] = padded_X[r_start:r_end, c_start:c_end, :]
        patches_labels[idx] = Y_2d[r, c]
    
    print(f"✅ 成功创建 {total_patches:,} 个patches")
    
    return patches_data, patches_labels
