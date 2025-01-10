import argparse
import os
import numpy as np
# from  util.mhd_util import read_mhd
from util.nii_util import read_nii
import util.h5_util as h5util
from scipy.ndimage import zoom



class Preprocessor():
    def __init__(
        self,
        data_dir,
        image_size,
        output_dir,
        subset,
        format = "npy",
        ext = 'nii.gz',
        save_name = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.new_shape=image_size
        self.output_dir = output_dir
        self.data_subset = subset
        assert format in ["npy", "h5"]
        self.format = format
        self.save_name = save_name if save_name is not None else self.data_dir.split('/')[-1]
        self.filenames = [p[:-(len(ext)+1)] for p in os.listdir(data_dir)]
        
        print(f'Num of cases found: {len(self.filenames)}')
        
        self.img_paths = []
        for p in self.filenames:
            self.img_paths.append(f"{data_dir}/{p}.{ext}")

    def _calculate_ratio_pads(self, img):
         # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:3]  # current shape [ height, width, depth]
        
        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1], self.new_shape[2] / shape[2])

        r = 1.0 if r >= 1.0 else r

        
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r)), int(round(shape[2] * r))
        dx = self.new_shape[0] - new_unpad[0]
        dy = self.new_shape[1] - new_unpad[1]
        dz = self.new_shape[2] - new_unpad[2]

        # divide padding into 2 sides
        lx, ly, lz = dx // 2, dy // 2, dz // 2
        rx, ry, rz = dx - lx, dy - ly, dz - lz

        pads = ((lx, rx), (ly, ry), (lz, rz))
        # print(r, pads, shape)
        return r, pads
    
    def _resize(self, img, r, pads, order):
        img = zoom(img, r, order=order)
        img = np.pad(img, pads)
        return img

    def process_data(self, index):
        path = self.img_paths[index]
        print(f'Processing {index}...')

        # Normalising range 
        # clipping HU range to [0, 2000] for OxAAA, for [-1000, 1000] referring to nnUNet pre processing
        (lower_b, upper_b) = (0, 2000)

        img = read_nii(path)
        print(f'Loaded shape: {img.shape}')
        
        r, pads = self._calculate_ratio_pads(img)
        img = self._resize(img, r, pads, 3)
        print(f'resezed shape: {img.shape}')
        # clipping HU range to [0, 2000] for OxAAA, for [-1000, 1000] referring to nnUNet pre processing
        img = np.clip(img, a_min=lower_b, a_max=upper_b)
        # normalise to [-1,1]
        img = ((img - lower_b) / (upper_b - lower_b) - 0.5) * 2.

        # as float32
        img = img.astype(np.float32)

        if self.format == "h5":
            output_filepath = f"{self.output_dir}/mitea.h5"

            img = np.expand_dims(img, axis=0)

            h5util.save(output_filepath, f"{self.data_subset}/{self.save_name}", img, dtype="float32")
            h5util.save_str(output_filepath, f"{self.data_subset}/filenames", self.filenames[index])
        else:
            img_dir = f"{self.output_dir}/{self.data_subset}/{self.save_name}"

            os.makedirs(img_dir, exist_ok=True)
            
            np.save(f"{img_dir}/{self.filenames[index]}.npy", img)    
            print(f'Processed saved to {img_dir}/{self.filenames[index]}.npy')        
      
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--format", type=str, default="npy")
    parser.add_argument("--save_name", type=str, default="image")
    args = parser.parse_args()

    data_dir = args.data_dir
    image_size = args.image_size
    output_dir = args.output_dir
    subset = args.subset
    format = args.format

    prepro = Preprocessor(
        data_dir=data_dir,
        image_size=image_size,
        output_dir=output_dir,
        subset=subset,
        format=format
    )

    for i in range(len(prepro.filenames)):
        print(f"Processing image {i+1}/{len(prepro.filenames)}", end="\r")
        prepro.process_data(i)