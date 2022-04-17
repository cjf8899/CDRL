import glob
import os
import cv2
from torch.utils.data import Dataset


class CDRL_Dataset(Dataset):
    def __init__(self, root_path=None, dataset=None, train_val=None, transforms_A=None, transforms_B=None):
        self.total_path = os.path.join(root_path, dataset, train_val)
        self.transforms_A = transforms_A
        self.transforms_B = transforms_B
        self.files = sorted(glob.glob(self.total_path + "/A/*.*")) +\
                      sorted(glob.glob(self.total_path + "/B/*.*"))
        self.A2BB2A_path = self.total_path.replace(dataset, dataset+'_A2B_B2A')

    def __getitem__(self, index):
        img_name = self.files[index % len(self.files)].split('/')[-1]
        
        img_A = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
        img_ori = img_A.copy()
        
        if '/A/' in self.files[index % len(self.files)]:
            img_B = cv2.imread(self.A2BB2A_path + '/A/'+img_name, cv2.IMREAD_COLOR)
        elif '/B/' in self.files[index % len(self.files)]:
            img_B = cv2.imread(self.A2BB2A_path + '/B/'+img_name, cv2.IMREAD_COLOR)
        
        transformed_A = self.transforms_A(image=img_A)
        transformed_B = self.transforms_B(image=img_B)
        
        img_A = transformed_A["image"]
        img_B = transformed_B["image"]
        
        return {"A":img_A , "B": img_B}

    def __len__(self):
        return len(self.files)


    


class CDRL_Dataset_test(Dataset):
    def __init__(self, root_path=None, dataset=None, transforms=None):
        self.total_path = os.path.join(root_path, dataset, 'val')
        self.transforms = transforms
        self.files = sorted(glob.glob(self.total_path + "/A/*.*"))
        
    def __getitem__(self, index):
        name = self.files[index % len(self.files)].split('/')[-1]
        
        img_A = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
        img_B = cv2.imread(self.files[index % len(self.files)].replace('/A/','/B/'), cv2.IMREAD_COLOR)
        
        transformed_A = self.transforms(image=img_A)
        transformed_B = self.transforms(image=img_B)
        
        img_A = transformed_A["image"]
        img_B = transformed_B["image"]
        
        return {"A": img_A, "B": img_B, 'NAME': name}

    def __len__(self):
        return len(self.files)
