import numpy as np
from torch.utils.data import Dataset
import nibabel as nib 
import os
import torch

class Dataset3D_NII(Dataset):
    def __init__(
        self,
        img_dir,
    ):
        super().__init__()
        self.img_dir = img_dir
        
        filenames = os.listdir(img_dir)
        self.filenames = [p for p in filenames if p.endswith(".nii.gz")]
        self.filenames.sort()


    def __len__(self):
        return len(self.filenames)

    def __read_image(self, path):
        img = nib.load(path)
        data = img.get_fdata()
        return data

    def __getitem__(self, index):
        # print(f'loading {self.img_dir}/{self.filenames[index]}')
        img = self.__read_image(f"{self.img_dir}/{self.filenames[index]}")
        

        # change to float
        img = img.astype(np.float32)


        item = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
        
        return item 

class Dataset3D_NPY(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        filenames = os.listdir(img_dir)
        self.filenames = [p for p in filenames if p.endswith(".npy")]
        self.filenames.sort()


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        # print(f'loading {self.img_dir}/{self.filenames[index]}')
        img = np.load(f"{self.img_dir}/{self.filenames[index]}")
        # label = np.load(f"{self.label_dir}/{self.filenames[index]}")
        

        # change to float
        img = img.astype(np.float32)
        # label = label.astype(np.float32)

        # # normalize to [0,1]
        # img = img / 255.0
        img = np.clip(img, 1000, 1256)
        # label = np.clip(label, 1000, 1256)
        img = (img - 1000) / 255. 
        # img = ((image - 1000) / 255 - 0.5) * 2.
        # label = (label - 1000) / 255. 
        # label = ((image - 1000) / 255 - 0.5) * 2.


        # img = np.stack((img, label), axis=0)
        item = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
        
        return item



if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    import time
    import matplotlib.pyplot as plt

    img_dir = r"..\data\data_mitea\train\images"
    label_dir = r"..\data\data_mitea\train\labels"

    image_size = (160, 160, 128)
    ds = Dataset3D_NPY(img_dir, label_dir)
    dl = DataLoader(ds, batch_size = 4, shuffle=False)

    plot_img = False
    iter = 1
    

    if not plot_img:
        start_time = time.time()
        for k in range(iter):
            for i, data in enumerate(dl):
                start_loop = time.time()
                img = data
                end_loop = time.time()
                print(f"{i+1} Loop time: {end_loop - start_loop}s", end="\r")

        
        print(f"Total: {(end_loop - start_time):.2f}s")
    else:
        for i, data in enumerate(dl):
            
            print(data.shape)
            
            slice_idx = image_size[0]//2

            # slice_idx = 73
            plt.subplot(321)
            plt.imshow(data[0, 0 , slice_idx, : , :])
            plt.subplot(322)
            plt.imshow(data[0, 1 , slice_idx, : , :])

            
            slice_idx = image_size[1]//2
            plt.subplot(323)
            plt.imshow(data[0, 0 , :, slice_idx, :])
            plt.subplot(324)
            plt.imshow(data[0, 1 , :, slice_idx, :])


            slice_idx = image_size[2]//2
            plt.subplot(325)
            plt.imshow(data[0, 0 , ..., slice_idx])
            plt.subplot(326)
            plt.imshow(data[0, 1 , ..., slice_idx])

            plt.show()
            break
        