import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

#----------------zigzag index generator-------------
def zigzag_coords(bsz):
    coords = []
    i = j = 0
    up = True
    for _ in range(bsz*bsz):
        coords.append((i,j))
        if up:
            if j == bsz-1:
                i += 1; up = False
            elif i == 0:
                j += 1; up = False
            else:
                i -= 1; j += 1
        else:
            if i == bsz-1:
                j += 1; up = True
            elif j == 0:
                i += 1; up = True
            else:
                i += 1; j -= 1
    return coords

zz4 = zigzag_coords(4)

def extract_zigzag_features(block16: np.ndarray, block_size = 4):
    feat = []
    H, W = block16.shape
    for row in range(0, H, block_size):
        for col in range(0, W, block_size):
            sub = block16[row:row+block_size, col:col+block_size]
            for (i,j) in zz4:
                feat.append(sub[i, j])
    return np.array(feat, dtype=sub.dtype)

def normalize_to_pi(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    denom = mx - mn
    if abs(denom) < eps:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mn) / denom) * np.pi

class AugmentedZigzagDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        imgs: np.ndarray | None = None,
        lbls: np.ndarray | None = None,
        npz_path: str | None = None,
        split: str | None = None,
    ):
        super().__init__()
        if imgs is not None and lbls is not None:
            self.imgs = imgs
            self.lbls = lbls
        # otherwise load from .npz
        elif npz_path is not None and split in ("train", "val", "test"):
            data = np.load(npz_path)
            if split == "train":
               self.imgs = data["train_images"]
               self.lbls = data["train_labels"].ravel() 
            elif split == "val":
               self.imgs = data["val_images"]
               self.lbls = data["val_labels"].ravel() 
            else:
               self.imgs = data["test_images"]
               self.lbls = data["test_labels"].ravel()
        else:
            raise ValueError("Must pass either (imgs & lbls) or (npz_path & split)")          
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    def __len__(self):
        return len(self.lbls)
    def __getitem__(self, idx):
        img_np = self.imgs[idx].astype(np.uint8)
        label = int(self.lbls[idx])
        img = Image.fromarray(img_np)
        img_t = self.transform(img)
        sub16 = F.adaptive_avg_pool2d(img_t.unsqueeze(0), (16,16))[0,0]
        feat_np = extract_zigzag_features(sub16.numpy())
        feat_np = normalize_to_pi(feat_np)
        x = torch.from_numpy(feat_np).float()
        y = torch.tensor(label, dtype = torch.long)
        return x, y
