from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np


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

  
  
def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
  dataset = copy.deepcopy(dataset)
  dataset.transform = expample_transform
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
