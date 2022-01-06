import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import re

from image_utils import show_cam_on_image, show_overlapped_cam


def rn_relevance(image, target_features, img_encoder, method, device, neg_saliency=False):   
  
    target_layers = [img_encoder.layer4[-1]]
    
    cam = method(model=img_encoder,
                  target_layers=target_layers,
                  use_cuda=torch.cuda.is_available())
    
    image_features = img_encoder(image)
    
    if neg_saliency:
        target_encoding = -target_features
    else:
        target_encoding = target_features
        
    image_relevance = cam(input_tensor=image, target_encoding=target_encoding)[0].squeeze()
    image_relevance = torch.FloatTensor(image_relevance)
   
    
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (1e-7+image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    
    return image_relevance, image



def interpret_rn(image, target_features, img_encoder, method, device, neg_saliency=False):   
   
    image_relevance, image = rn_relevance(image, target_features, img_encoder, method, device, neg_saliency=neg_saliency)
    vis = show_cam_on_image(image, image_relevance, neg_saliency=neg_saliency)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.imshow(vis)
    
    
    
def interpret_rn_overlapped(image, target_features, img_encoder, method, device):   
   
    pos_image_relevance, _ = rn_relevance(image, target_features, img_encoder, method, device, neg_saliency=False)
    neg_image_relevance, image = rn_relevance(image, target_features, img_encoder, method, device, neg_saliency=True)

    vis = show_overlapped_cam(image, neg_image_relevance, pos_image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.imshow(vis)
    
    
    
def rn_perword_relevance(image, text, clip_model, clip_tokenizer, method, device, masked_word=""):   
    
    clip_model.eval()
    
    main_text = clip_tokenizer(text).to(device)
    # remove the word for which you want to visualize the saliency
    masked_text = re.sub(masked_word, "", text)
    masked_text= clip_tokenizer(masked_text).to(device)
    
    image_features = clip_model.encode_image(image)
    main_text_features = clip_model.encode_text(main_text)
    masked_text_features = clip_model.encode_text(masked_text)
    
    image_features_norm = image_features.norm(dim=-1, keepdim=True)
    image_features_new = image_features / image_features_norm
    main_text_features_norm = main_text_features.norm(dim=-1, keepdim=True)
    main_text_features_new = main_text_features / main_text_features_norm
    
    masked_text_features_norm = masked_text_features.norm(dim=-1, keepdim=True)
    masked_text_features_new = masked_text_features / masked_text_features_norm
  
    target_encoding = main_text_features_new-masked_text_features_new
  
    target_layers = [clip_model.visual.layer4[-1]]
    
    cam = method(model=clip_model.visual,
                  target_layers=target_layers,
                  use_cuda=torch.cuda.is_available())
    
    image_features = clip_model.visual(image)
   
        
    image_relevance = cam(input_tensor=image, target_encoding=target_encoding)[0].squeeze()
    image_relevance = torch.FloatTensor(image_relevance)
   
    
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (1e-7+image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    
    return image_relevance, image



def interpret_perword_rn(image, text, clip_model, clip_tokenizer, method, device, masked_word=""):   
   
    image_relevance, image = rn_perword_relevance(image, text, clip_model, clip_tokenizer, method, device, masked_word)
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    plt.imshow(vis)