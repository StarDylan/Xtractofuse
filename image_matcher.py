import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet, viz2d
from lightglue.utils import load_image, rbd, match_pair
import torch
from skimage.color import rgb2gray
from skimage.io import imread



def matcher(image):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device) # load the extractor
    matcher = LightGlue(features='superpoint').eval().to(device)  # load the matcher

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = load_image('baker1.jpg').to(device)
    image1 = load_image('baker2.jpg').to(device)

    # extract local features
    feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)

    # match the features
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
    points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

    # feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)

    print(image0.shape)
    print(image1.shape)
    

    axes = viz2d.plot_images([image0, image1])
    # viz2d.plot_matches(points0, points1, color="lime", lw=0.2)
    # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

    
    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    # viz2d.plot_images([image0, image1])
    # viz2d.plot_keypoints([points0, points1], colors=[kpc0, kpc1], ps=10)

matcher(None)