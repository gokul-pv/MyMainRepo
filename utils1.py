import torch
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import albumentations as A
import albumentations.pytorch as AP
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

class AlbumentationTransforms:

  """
  Helper class to create test and train transforms using Albumentations
  """

  def __init__(self, transforms_list=[]):
    transforms_list.append(AP.ToTensorV2())
    self.transforms = A.Compose(transforms_list)


  def __call__(self, img):
    img = np.array(img)
    #print(img)
    return self.transforms(image=img)['image']

  
  
def visualize_augmentations(dataset,transforms, idx=0, samples=10, cols=5 ):
  dataset = copy.deepcopy(dataset)
  dataset.transform = transforms
  rows = samples // cols
  figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
  for i in range(samples):
      image, _ = dataset[idx]
      # image = image / 2 + 0.5     # unnormalize
      image = image * STD[:, None, None] + MEAN[:, None, None]
      # plt.imshow(x.numpy().transpose(1, 2, 0))
      ax.ravel()[i].imshow(np.transpose(image, (1, 2, 0)))
      ax.ravel()[i].set_axis_off()
  plt.tight_layout()
  plt.show()

  

def imshow(img,c ):
#   img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  fig = plt.figure(figsize=(7,7))
  plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
  plt.title(c)


def show_train_data(dataset, classes):	
  dataiter = iter(dataset)
  images, labels = dataiter.next()
  for i in range(10):
    index = [j for j in range(len(labels)) if labels[j] == i]
    imshow(torchvision.utils.make_grid(images[index[0:5]],nrow=5,padding=2,scale_each=True),classes[i])  
  
  

def evaluate_accuracy(model, device, testloader,misclassified_images, classes,correct_pred, total_pred):
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            for i in range(len(predictions)):
                if predictions[i]!= labels[i]:
                    misclassified_images.append([images[i], predictions[i], labels[i]])

		
		
def show_misclassified_images(misclassified_images, classes, correct_pred, total_pred): 
	fig = plt.figure(figsize = (10,10))
	for i in range(10):
	  sub = fig.add_subplot(5, 2, i+1)
	  img = misclassified_images[i][0].cpu()
	  img = img * STD[:, None, None] + MEAN[:, None, None]
	  # img = img / 2 + 0.5 
	  npimg = img.numpy()
	  plt.imshow(np.transpose(npimg,(1, 2, 0)),interpolation='none')

	  sub.set_title("Pred={}, Act={}".format(str(classes[misclassified_images[i][1].data.cpu().numpy()]),
						 str(classes[misclassified_images[i][2].data.cpu().numpy()])))

	plt.tight_layout()
	plt.show()

	# print accuracy for each class
	for classname, correct_count in correct_pred.items():
	    accuracy = 100 * float(correct_count) / total_pred[classname]
	    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))
		
		
		
def plot_loss_accurracy(train_losses,train_accuracy, test_losses,  test_accuracy):
	fig, axs = plt.subplots(2,2,figsize=(15,12))
	axs[0, 0].plot(train_losses, label='Training Loss')
	axs[0, 0].grid(linestyle='-.')
	axs[0, 0].set_title("Training Loss")
	axs[0, 0].legend()

	axs[1, 0].plot(train_accuracy, label='Training Accuracy')
	axs[1, 0].grid(linestyle='-.')
	axs[1, 0].set_title("Training Accuracy")
	axs[1, 0].legend()

	axs[0, 1].plot(test_losses, label='Test Loss')
	axs[0, 1].grid(linestyle='-.')
	axs[0, 1].set_title("Test Loss")
	axs[0, 1].legend()

	axs[1, 1].plot(test_accuracy, label='Test Accuracy')
	axs[1, 1].grid(linestyle='-.')
	axs[1, 1].set_title("Test Accuracy")
	axs[1, 1].legend()		

	
	
import torch
import torch.nn.functional as F
class GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    A simple example:
        # initialize a model, model_dict and gradcam
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, model, layer_name):
        self.model = model
        # self.layer_name = layer_name
        self.target_layer = layer_name

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def saliency_map_size(self, *input_size):
        device = next(self.model.parameters()).device
        self.model(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        
        self.gradients.clear()
        self.activations.clear()
        return saliency_map, logit
    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)





# ------------------------------------VISUALIZE_GRADCAM-------------------------------------------------------------


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result

#-------------------------------------------GradCam View (Initialisation)--------------------------------------------



def GradCamView(miscalssified_images,model,classes,layers,Figsize = (23,30),subplotx1 = 9, subplotx2 = 3):

    fig = plt.figure(figsize=Figsize)
    for i,k in enumerate(miscalssified_images):
        images1 = [miscalssified_images[i][0].cpu()* STD[:, None, None]+MEAN[:, None, None]]
        images2 =  [miscalssified_images[i][0].cpu()* STD[:, None, None]+MEAN[:, None, None]]
        for j in layers:
                g = GradCAM(model,j)
                mask, _= g(miscalssified_images[i][0].clone().unsqueeze_(0))
                heatmap, result = visualize_cam(mask,miscalssified_images[i][0].clone().unsqueeze_(0)* STD[:, None, None]+MEAN[:, None, None] )
                images1.extend([heatmap])
                images2.extend([result])
        # Ploting the images one by one
        grid_image = make_grid(images1+images2,nrow=len(layers)+1,pad_value=1)
        npimg = grid_image.numpy()
        sub = fig.add_subplot(subplotx1, subplotx2, i+1) 
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        sub.set_title('P = '+classes[int(miscalssified_images[i][1])]+" A = "+classes[int(miscalssified_images[i][2])],fontweight="bold",fontsize=18)
        sub.axis("off")
        plt.tight_layout()
        fig.subplots_adjust(wspace=0)

