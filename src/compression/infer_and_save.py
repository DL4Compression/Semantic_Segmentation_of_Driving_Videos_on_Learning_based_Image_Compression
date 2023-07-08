from bz2 import compress
import torch, torchvision, os
from torch.utils import data
from torch.backends import cudnn
cudnn.benchmark = True
import torch.nn as nn
import numpy as np
import sys
import pickle, pdb
from PIL import Image
from dahuffman import HuffmanCodec
import cv2
import time
from tqdm import tqdm as tq
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

from utils.model import autoencoder
from utils.dataloader_test import IDDCOM
import timeit

# The Float->Int module
class Float2Int(nn.Module):
    def __init__(self, bit_depth=8):
        super().__init__()
        self.bit_depth = bit_depth

    def forward(self, x):
        x = torch.round(x * (2**self.bit_depth - 1)).type(torch.int32)
        return x

# The Int->Float module
class Int2Float(nn.Module):
    def __init__(self, bit_depth=8):
        super().__init__()
        self.bit_depth = bit_depth

    def forward(self, x):
        x = x.type(torch.float32) / (2**self.bit_depth - 1)
        return x

def main( args ):
    # module instances
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    model = autoencoder(args.n_convblocks)
    float2int = Float2Int(args.bit_depth)
    int2float = Int2Float(args.bit_depth)

    # GPU transfer
    

    # load the given model
    loaded_model_file = torch.load(args.model_file, map_location = torch.device('cpu'))
    model.load_state_dict(loaded_model_file['model_state'])
    if torch.cuda.is_available() and args.gpu:
        # print("model sent to cuda")
        model = model.to(device)
        float2int, int2float = float2int.to(device), int2float.to(device)
    
    images_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    labels_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    avg_cf, avg_ssim, avg_psnr,avg_bits = 0., 0., 0., 0.
    n = 0
    
    for idx,dirc in enumerate(tq(os.listdir(args.inferdata))):
        
        bit_rate_raw = []
        bit_rate_cae = []
        inf_path = os.path.join(args.inferdata,dirc)
        iddinfer = IDDCOM(inf_path, transform_images=images_transforms, transform_masks=labels_transforms, preserve_names=True)
        iddinferdl = data.DataLoader(iddinfer, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

        model.eval() # switch to evaluation mode
        with torch.no_grad():
            # a global counter & accumulation variables
            
            all_cf = []
            all_bits = []
            
            for idx, (image, _, name) in enumerate(tq(iddinferdl)):
                img_size = sys.getsizeof(image.storage())
                if torch.cuda.is_available() and args.gpu:
                    image = image.to(device)

                start = timeit.timeit()
                compressed = model.encoder(image) # forward through encoder
                end = timeit.timeit()
                latent_int = float2int(compressed) # forward through Float2Int module

                # usual numpy conversions
                image_numpy = image.cpu().numpy()
                latent_int_numpy = latent_int.cpu().numpy()
                latent_inp = torch.tensor(latent_int_numpy)
                if args.with_aac:
                    # encode latent_int with Huffman coding
                    inpt=[]
                    c1, c, h, w = latent_int_numpy.shape
                    flat = latent_int_numpy.flatten()
                    for i in flat:
                        inpt.append(str(i))
                    codec = HuffmanCodec.from_data(inpt)
                    encoded = codec.encode(inpt)
                    hufbook = codec.get_code_table()
                    book_size = sys.getsizeof(hufbook)
                    code_size = sys.getsizeof(encoded)
                    
                    bit_rate_raw.append(img_size*(30/1.8))
                    bit_rate_cae.append((book_size+code_size)*(30/1.8))

                    cf = img_size/(book_size+code_size)
                    all_cf.append(cf)
                    bits_al = []
                    for symbol, (bits, val) in hufbook.items():
                        bits_al.append(bits)
                    bits_al = np.array(bits_al)
                    av_bits = np.mean(bits_al)
                    all_bits.append(av_bits)
                    decoded = codec.decode(encoded)
                    ar_in=[]
                    for i in decoded:
                        ar_in.append(int(i))
                    ar_in = np.array(ar_in)
                    latent = ar_in.reshape([c1,c,h,w])
                    latent_inp = torch.from_numpy(latent).cuda()
                else:
                    bits = args.bit_depth
                    Q = None
                # st= timeit.timeit()
                latent_float = int2float(latent_inp) # back to float
                decompressed = model.decoder(latent_float) # forward through decoder

                original, reconstructed = image_numpy, decompressed.cpu().numpy()
                # en = timeit.timeit()
                all_cf.append(cf)

                n += 1

                if args.produce_latent_code:
                    # save the latent code if requested. the saved items are
                    if not(os.path.exists(os.path.join(args.out_latent,dirc))):
                        os.mkdir(os.path.join(args.out_latent,dirc))
                    np.save(os.path.join(args.out_latent,dirc)+'/'+name[0],latent_float.detach().cpu())
                    

                if args.produce_decompressed_image:
                    if not(os.path.exists(os.path.join(args.out_decom,dirc))):
                        os.mkdir(os.path.join(args.out_decom,dirc))
                    reconstructed = reconstructed.squeeze()
                    reconstructed = np.asarray(reconstructed*255, dtype=np.uint8)
                    reconstructed = reconstructed.transpose(1,2,0)
                    reconstructed = Image.fromarray((reconstructed).astype(np.uint8))
                    decom_file = os.path.join(os.path.abspath(args.out_decom),dirc,str(args.bit_depth) + '_decom_' + name[0])
                    reconstructed.save(decom_file)

                if n == args.max_samples:
                    break
            if(n==args.max_samples):
                break
        raw_bitrate = np.array(bit_rate_raw)
        cae_bitrate = np.array(bit_rate_cae)
        os.mkdir(os.path.join(args.bit_dir,dirc))
        np.save(os.path.join(args.bit_dir,dirc,'raw_bitrate'),raw_bitrate)
        np.save(os.path.join(args.bit_dir,dirc,'cae_bitrate'),cae_bitrate)
        
    avg_cf = np.mean(all_cf)
    avg_bits = np.mean(all_bits)
    import json
    json_content = []
    json_fillpath = os.path.abspath(args.plot_json)
    if not os.path.exists(json_fillpath):
        with open(json_fillpath, 'w') as json_file:
            json.dump([], json_file)

    with open(json_fillpath, 'r') as json_file:
        json_content = json.load(json_file)

    # append to the content of json
    json_content.append({'d': args.n_convblocks,'bit_length':args.bit_depth,'avg_bits':avg_bits,'avg_cf': avg_cf,'Model_name':args.model_file})

    with open(json_fillpath, 'w') as json_file:
        json.dump(json_content, json_file)

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser("""
	The inference script works on a folder level. Provide this script a folder (--inferdata) with full-scale mammograms which can be created with the dataprep.py script.
	It also requires a model (--model_file) file to work on which is to be produced by train.py script.
	All the full-scale mammograms will be processed on by one to collect bpp, ssim and psnr information which will be averaged to produce the numbers to be plotted.
	
	To produce bpp-ssim-psnr with varying bit-depth, run the over and over JUST with different --bit_depth. Keep the json filename same.

	(Optionally) Provide a json filename to write the metrics to.
	(Optionally) You may choose to use AAC (Huffman coding).
	(Optionally) You may choose to use GPU for all processing.
	(Optionally) You may choose to produce (--produce_latent_code and --out_latent) the latent code (integer) and the corresponding huffman codebook.
	(Optionally) You may choose to produce (--produce_decompressed_image and --out_decom) the decompressed images for visualization purpose.
	(Optionally) You may choose the number of Convolution blocks in the architecture. (CVPR paper have 3)
		""")
	parser.add_argument('-m', '--model_file', required=True, type=str, help='Path to the model file')
	parser.add_argument('--inferdata', type=str, required=True, help='Path to the folder containing full scale mammogram images for inference')
	parser.add_argument('-t', '--bit_depth', required=True, type=int, help='Required bit depth for Float2Int quantization')
	parser.add_argument('--gpu', action='store_true', help='Want GPU ?')
	parser.add_argument('--with_aac', action='store_true', help='Use Adaptive Arithmatic Coding (Huffman coding)')
	parser.add_argument('-l', '--produce_latent_code', action='store_true', help='Write latent code tensor (integer) as output (possibly along huffman codebook)')
	parser.add_argument('-d', '--produce_decompressed_image', action='store_true', help='Write decompressed images as output')
	parser.add_argument('--out_latent', type=str, required=False, default='.', help='Folder to produce the latent codes into ?')
	parser.add_argument('--out_decom', type=str, required=False, default='.', help='Folder to produce decompressed images into ?')
	parser.add_argument('-p', '--plot_json', type=str, required=False, default='./plot.json', help='Path for the output json file')
	parser.add_argument('--n_convblocks', type=int, required=False, default=3, help='no of conv blocks in the architecture')
	parser.add_argument('-x', '--max_samples', type=int, required=False, default=0, help='limit the number of samples to use for inference')
	parser.add_argument('--bit_dir' ,type=str,required =False,default='.' ,help='bits array store ?')
	args = parser.parse_args()

	main( args )
