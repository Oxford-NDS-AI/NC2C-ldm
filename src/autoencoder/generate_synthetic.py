import h5py
import torch
from autoencoder import AutoencoderKL
import matplotlib.pyplot as plt
import numpy as np
import util.h5_util as h5util
import time
import os
from torchvision.utils import make_grid
import argparse
import yaml

def sanitize_img(x):
    """
    Convert the given image to a numpy array
    """
    if np.min(x) < 0:
        x += np.min(x)
    np.clip(x, 0, 1, out=x)
    return x

def denormalize(img, im_min, im_max):
    return img * (im_max - im_min) + im_min

def stack_to_grid(echo_imgs, row, is_img=True):
    echo_imgs = np.transpose(echo_imgs, (0, 3, 1,2))
    echo_imgs = torch.from_numpy(echo_imgs).type(torch.FloatTensor)
    grid = make_grid(echo_imgs, padding=0, nrow=row)
    grid = grid.cpu().numpy()
    if is_img:
        np.clip(grid, 0, 255, out=grid)
    else:
        # just for visualization
        grid = grid / np.max(grid) * 255 
        
    grid = np.transpose(grid, (1,2,0)).astype(np.uint8)
    grid = grid[:,:,0].squeeze()
    return grid

def update_list(echo_imgs, echo_img, max_count=96):
    if echo_imgs is None:
        echo_imgs = echo_img
    else:
        if echo_imgs.shape[0] < max_count:
            echo_imgs = np.concatenate((echo_imgs, echo_img), axis=0)
    return echo_imgs

def show_images(echo_imgs, label_imgs, max_count=96):
    nr = 8
    grid1 = stack_to_grid(echo_imgs[:max_count], row=nr)
    grid2 = stack_to_grid(label_imgs[:max_count], row=nr, is_img=False)

    plt.subplot(121)
    plt.imshow(grid1)
    plt.subplot(122)
    plt.imshow(grid2)
    plt.show()

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description='Generate synthetic images')
    parser.add_argument('--model-dir', type=str, required=True, help='Path to the models dir')
    parser.add_argument('--input', type=str, required=True, help='Path to the H5 samples file')
    parser.add_argument('--n-data', type=int, default=0, help='Number of data to generate')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output dir')
    args = parser.parse_args()

    model_dir = args.model_dir
    input_file = args.input
    nr_data = args.n_data
    output_dir = args.output_dir

    config_filepath = f"{model_dir}/vae_params.yaml"

    # get the filename from the input path
    filename = os.path.basename(input_file)
    output_file = f"{output_dir}/_synthetic{filename}"
    colname = f"samples"
    output_imgcol = f"images"
    output_labelcol = f"labels"

    img_size = (160, 160, 128)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # config stuff
    with open(config_filepath, 'r') as config_file:
        configs = yaml.safe_load(config_file)

    ddconfig = configs['ddconfig']
    lossconfig = configs['lossconfig']
    embed_dim = configs['hyperparams']['embed_dim']
    
    network = AutoencoderKL(ddconfig, lossconfig, embed_dim, None, None, batch_size=1).to(device)

    # we need to get the global min/max to denormalize
    print(f"reading from {input_file}")
    with h5py.File(input_file, 'r') as hl:
        im_min = hl.get("min")[0]
        im_max = hl.get("max")[0]
        model_id = hl.get("model_id")[0]
        len_data = len(hl.get(colname))
    
    if nr_data == 0:
        nr_data = len_data
    
    # checkpoint
    version_dir = f"{model_dir}/checkpoints" 
    ckpt_files = os.listdir(version_dir)
    ckpt_files = [f for f in ckpt_files if f.startswith("epoch=0")]
    # sort and take last one
    ckpt_files.sort()
    CKPT_PATH = rf"{version_dir}/{ckpt_files[-1]}"

    print(f"Restoring from {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH)
    print(checkpoint.keys())

    network.load_state_dict(checkpoint['state_dict'], strict=False)
    network.eval()

    img_idx = 0
    echo_imgs = None
    label_imgs = None

    with torch.no_grad():
        start_time0 = time.time()
        for idx in range(0, nr_data):
            message = f"\rProcessing {idx+1} / {len_data}"

            start_time = time.time()
            img = h5util.load_img(input_file, colname, idx)
            img = denormalize(img, im_min, im_max)

            z = torch.from_numpy(img).type(torch.FloatTensor)
            
            recon = network.decode(z.to(device))
            recon = recon.squeeze().cpu().numpy()
            print(recon.shape)
        
            plt.subplot(121)
            plt.imshow(recon[..., 80], cmap='viridis')
            plt.colorbar()
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(recon[..., 80], cmap='viridis')
            plt.colorbar()
            plt.axis('off')
            
            plt.savefig(f"{output_dir}/echo-{idx}.png")
            plt.close()

            # recon = np.transpose(recon, (0,2,3,4,1))
            
            # recon_img = sanitize_img(recon[...,0])
            # recon_img = recon_img * 255
            # print(recon_img.shape)

            # # recon_label = np.argmax(recon[...,1:], axis=-1)

            # # convert to uint8
            # # recon_img = recon_img.astype(np.uint8)
            # # recon_label = recon_label.astype(np.uint8)
        
            # recon = np.stack([recon_img, recon_label], axis=-1)

            # h5util.save(output_file, output_imgcol, recon_img, compression="gzip", dtype="uint8")
            # # h5util.save(output_file, output_labelcol, recon_label, compression="gzip", dtype="uint8")

            # # For visualization purposes
            # echo_img  = recon[:,img_size[0]//2,..., :1]
            # label_img = recon[:,img_size[0]//2,..., 1:]

            

            # echo_imgs = update_list(echo_imgs, echo_img)
            # label_imgs = update_list(label_imgs, label_img)
            
            # message +=(f" {(time.time()-start_time):.2f} sec")
            # print(f"\r{message}", end='')

            
        # end of for loop
        print(f"\n{len_data} images generated! Total time: {(time.time()-start_time0):.2f} sec")
    # end of torch no grad

    # show_images(echo_imgs, label_imgs)




