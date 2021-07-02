import torch
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import numpy as np
import copy
import matplotlib.pyplot as plt

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
  MEAN = torch.tensor([0.485, 0.456, 0.406])
  STD = torch.tensor([0.229, 0.224, 0.225])
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

  
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


# functions to show an image
def imshow(img,c ):
#   img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  fig = plt.figure(figsize=(7,7))
  plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
  plt.title(c)


def show_train_data(dataset, classes):

	# get some random training images
  dataiter = iter(dataset)
  images, labels = dataiter.next()
  for i in range(10):
    index = [j for j in range(len(labels)) if labels[j] == i]
    imshow(torchvision.utils.make_grid(images[index[0:5]],nrow=5,padding=2,scale_each=True),classes[i])  
  
  
