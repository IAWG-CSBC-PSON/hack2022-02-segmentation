import argparse
import os
import tensorflow as tf
import numpy as np
import sys

from pathlib import Path
from skimage import io

from toolbox import GPUselect


def main(imagePath, outputPath):
    """
    """
    model_path = Path(__file__).parent.joinpath('models', 'artefact', 'artefact_segmentation.h5')

    model = tf.keras.models.load_model(model_path)

    iname = imagePath.split('/')[-1]

    img = io.imread(imagePath)

    if img.ndim == 2 and img.shape == (256, 256):
        ch1 = np.zeros([256, 256], dtype=np.uint8)
        ch2 = np.zeros([256, 256], dtype=np.uint8)
        img = np.dstack([img, ch1, ch2])

    if img.ndim == 3:
        if img.shape[0] == 2:
            np.reshape(img, (1, 2, 0))
        if img.shape[2] == 2:
            ch = np.zeros([256, 256], dtype=np.uint8)
            img = np.dstack([img, ch1, ch2])

    predictions = model.predict(np.expand_dim(img, 0))[0]

    pred_b = np.zeros_like(predictions)
    pred_b[np.arange(len(predictions)), predictions.argmax(1)] = 1
    pred = (pred_b[:, :, 0] == 0).astype(int)

    img_masks = io.imread('/probability_maps/%_NucleiPM_1.tif' % (iname.rsplit('.', 1)[0]))

    rval = np.multiply(pred, img_masks)

    io.imsave( outputPath, rval,  )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("imagePath", help="path to the .tif file")
    parser.add_argument("--model",  help="type of model. For example, nuclei vs cytoplasm",default = 'nucleiDAPILAMIN')
    parser.add_argument("--outputPath", help="output path of probability map")
    parser.add_argument("--channel", help="channel to perform inference on",  nargs = '+', default=[0])
    # parser.add_argument("--channel2", help="channel2 to perform inference on", type=int, default=-1)
    parser.add_argument("--classOrder", help="background, contours, foreground", type = int, nargs = '+', default=-1)
    parser.add_argument("--mean", help="mean intensity of input image. Use -1 to use model", type=float, default=-1)
    parser.add_argument("--std", help="mean standard deviation of input image. Use -1 to use model", type=float, default=-1)
    parser.add_argument("--scalingFactor", help="factor by which to increase/decrease image size by", type=float,
                        default=1)
    parser.add_argument("--stackOutput", help="save probability maps as separate files", action='store_true')
    parser.add_argument("--GPU", help="explicitly select GPU", type=int, default = -1)
    parser.add_argument("--outlier", help="map percentile intensity to max when rescaling intensity values. Max intensity as default", type=float, default=-1)
    args = parser.parse_args()

    logPath = ''
    scriptPath = os.path.dirname(os.path.realpath(__file__))
    modelPath = os.path.join(scriptPath, 'models', args.model)
    # modelPath = os.path.join(scriptPath, 'models/cytoplasmINcell')
    # modelPath = os.path.join(scriptPath, 'cytoplasmZeissNikon')
    pmPath = ''
    if os.system('nvidia-smi') == 0:
        if args.GPU == -1:
            print("automatically choosing GPU")
            GPU = GPUselect.pick_gpu_lowest_memory()
        else:
            GPU = args.GPU
        print('Using GPU ' + str(GPU))

    else:
        if sys.platform == 'win32':  # only 1 gpu on windows
            if args.GPU==-1:
                GPU = 0
                print('using default GPU')
            else:
                GPU = args.GPU
            print('Using GPU ' + str(GPU))
        else:
            GPU=0
            print('Using CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % GPU
    main(args.imagePath, args.outputPath)
