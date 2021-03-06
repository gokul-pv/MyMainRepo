'''Train CIFAR10 with PyTorch.'''

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def load_cifar(train_transform,test_transform,batch_size):
	

	#Get the Train and Test Set
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)


	SEED = 1

	# CUDA?
	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
			torch.cuda.manual_seed(SEED)

	# dataloader arguments - something you'll fetch these from cmdprmt
	dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

	trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
	testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

	classes = ('plane', 'car', 'bird', 'cat',
    	       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return classes, trainloader, testloader

def load(train_dataset,test_dataset,batch_size):		

	SEED = 1

	# CUDA?
	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
	    torch.cuda.manual_seed(SEED)

	# dataloader arguments - something you'll fetch these from cmdprmt
	dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

	trainloader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
	testloader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
	return trainloader, testloader


#Training & Testing Loops

def train(model, device, train_loader, optimizer, scheduler,criterion, epoch, train_losses, train_accuracy):
  model.train()
  
  correct = 0
  processed = 0
  train_loss = 0

  pbar = tqdm(train_loader)
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)
    
    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_loss += loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()
	
    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_loss /= len(train_loader.dataset)    
  train_losses.append(train_loss)
  train_accuracy.append(100. * correct / len(train_loader.dataset)) 

def test(model, device, criterion, test_loader,test_losses, test_accuracy ):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracy.append(100. * correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def train_test_loop(model, device, trainloader, testloader,criterion,optimizer, train_losses, train_accuracy, test_losses, test_accuracy,EPOCHS):
	for epoch in range(EPOCHS):
	    print("EPOCH:", epoch+1, 'LR:',optimizer.param_groups[0]['lr'])
	    train(model, device, trainloader, optimizer, criterion, epoch,train_losses,train_accuracy )
	    # scheduler.step()
	    test(model, device, criterion, testloader,test_losses, test_accuracy )

