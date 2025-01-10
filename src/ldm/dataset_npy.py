from torch.utils.data import Dataset
import torch
import numpy as np

class DatasetNpy(Dataset):
    def __init__(
        self,
        case_list_path,
    ):
        super().__init__()
        self.case_list_path = case_list_path
        self.all_case = np.load(self.case_list_path, allow_pickle=True)
        
        self.len_data = len(self.all_case)
        print(f"{self.len_data = }")

    def __len__(self):
        return self.len_data


    def __getitem__(self, index):
        (nc_pth, ce_pth) = self.all_case[index]
        nc_img = np.load(nc_pth)
        ce_img = np.load(ce_pth)
            
        assert nc_img.min() >= -1 and nc_img.max() <= 1, f'NC Image need to normalised to [-1,1] {nc_pth} [{nc_img.min()}, {nc_img.max()}]'
        assert ce_img.min() >= -1 and ce_img.max() <= 1, f'CE Image need to normalised to [-1,1] {ce_pth} [{ce_img.min()}, {ce_img.max()}]'

        
        # item = {
        #     'nc': torch.from_numpy(nc_img).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0), # unsqueeze to add c = 1 [c, d, h, w]
        #     'ce': torch.from_numpy(ce_img).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0)
        # }

        item = torch.from_numpy(nc_img).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0) # for testing DDPM
        return item