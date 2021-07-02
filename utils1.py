import torch
import torchvision
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
  
  

def evaluate_accuracy(model, device, test_loader,misclassified_images, correct_pred, total_pred):
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
	
	
	MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])
  
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
