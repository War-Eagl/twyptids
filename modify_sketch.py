import argparse
import os
import clip
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
from svgpathtools import svg2paths
import stroke_relevance

pydiffvg.set_print_timing(False)

gamma = 1.0


#parser = argparse.ArgumentParser(description = 'Modify the SVG sketches using text prompts')
#parser.add_argument("svg_path", type = str, nargs=1, help='path of the svg file to be modified')
#parser.add_argument("prompt", type = str, nargs=1, help='The prompt for modificatin')

#args = parser.parse_args()

path = '/home/ubuntu/scratch/twptids/original_sketches/camel.svg'
prompt = 'camel on fire'

folder_path = 'images/' + prompt
os.mkdir(folder_path)

#initialize clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device, jit=False)



# initialize pydiffvg and pytorch to GPU
text_input = clip.tokenize(prompt).to(device)
use_normalized_clip = False 
pydiffvg.set_use_gpu(torch.cuda.is_available())
device = torch.device('cuda')
pydiffvg.set_device(device)

# Optimization arguments
num_iter = 1000
canvas_width, canvas_height = 224,224

# encode the text prompt
with torch.no_grad():
    text_features = model.encode_text(text_input)

# Image Augmentation Transformation
augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
])

if use_normalized_clip:
    augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# initialize strokes
shapes = []
shape_groups = []
_,svg_paths = svg2paths(path)
for svg_path in svg_paths:
  svg_path = pydiffvg.from_svg_path(svg_path['d'])
  svg_path = svg_path[0]
  svg_path.stroke_width =  torch.tensor(1.0)
  shapes.append(svg_path)
  path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]))
  shape_groups.append(path_group)


# Just some diffvg setup
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
render = pydiffvg.RenderFunction.apply
img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
points_vars = []
for path in shapes:
    path.points.requires_grad = True
    points_vars.append(path.points)

params = []

relevances = stroke_relevance.calculate_stroke_importance(path,prompt)

for i, point_var in enumerate(points_vars):
    param ={'params': point_var, 'lr': relevances[i]}
    params.append(param)



# Optimizers
points_optim = torch.optim.Adam(params)


# Run the main optimization loop
for t in range(num_iter):

    # Anneal learning rate (makes videos look cleaner)
    if t == int(num_iter * 0.5):
        for g in points_optim.param_groups:
            g['lr'] = 0.4
    if t == int(num_iter * 0.75):
        for g in points_optim.param_groups:
            g['lr'] = 0.1
    
    points_optim.zero_grad()
   
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
    if t % 5 == 0:
        pydiffvg.imwrite(img.cpu(), folder_path +'/iter_{}.png'.format(int(t/5)), gamma=gamma)
    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW

    loss = 0
    NUM_AUGS = 4
    img_augs = []
    for n in range(NUM_AUGS):
        img_augs.append(augment_trans(img))
    im_batch = torch.cat(img_augs)
    image_features = model.encode_image(im_batch)
    for n in range(NUM_AUGS):
        loss -= torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)


    # Backpropagate the gradients.
    loss.backward()

    # Take a gradient descent step.
    points_optim.step()
