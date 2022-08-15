import torch
import CLIP.clip as clip
import numpy as np
import matplotlib.pyplot as plt
import pydiffvg
import torchvision.transforms as T
from pathlib import Path
from svgpathtools import svg2paths


start_layer =  -1
start_layer_text =  -1

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
      # calculate index of last layer 
      start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1: 
      # calculate index of last layer 
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text
   
    return text_relevance, image_relevance


def calculate_stroke_importance(svg_path, prompt, add_stroke = False):
  svg_path = Path(svg_path)
  file_name = svg_path.stem
  _,paths = svg2paths(svg_path)
  raster_path = '/content/out/{}.png'.format(file_name)

  shapes = []
  shape_groups = []
  for path in paths:
    path = pydiffvg.from_svg_path(path['d'])
    path = path[0]
    path.stroke_width =  torch.tensor(1.0)
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]))
    shape_groups.append(path_group)

  scene_args = pydiffvg.RenderFunction.serialize_scene(\
        224, 224, shapes, shape_groups)
  render = pydiffvg.RenderFunction.apply
  img = render(224, 224, 2, 2, 0, None, *scene_args)
  img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])

  img = img.detach().cpu()
  img = img.permute(2,0,1)
  print(img.shape)
  transform = T.ToPILImage()
  img = transform(img)
  


  img_path = raster_path
  img = preprocess(img).unsqueeze(0).to(device)
  texts = [prompt]
  text = clip.tokenize(texts).to(device)

  R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
  dim = int(R_image.numel() ** 0.5)
  image_relevance = R_image.reshape(1, 1, dim, dim)
  image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
  image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
  image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
  

  if add_stroke:
    return image_relevance

  
  shapes = []
  shape_groups = []
  stroke_relevances = []

  for path in paths:
    path = pydiffvg.from_svg_path(path['d'])
    path = path[0]
    path.stroke_width =  torch.tensor(1.0)
    shapes = [path]
    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]), fill_color = None, stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]))
    shape_groups = [path_group]
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        224, 224, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(224, 224, 2, 2, 0, None, *scene_args)
    img = img.detach().cpu().numpy()

    img = np.multiply(img[:,:,3], image_relevance)

    stroke_relevance = np.max(img)  
    stroke_relevances.append(stroke_relevance)  
  
  return stroke_relevances