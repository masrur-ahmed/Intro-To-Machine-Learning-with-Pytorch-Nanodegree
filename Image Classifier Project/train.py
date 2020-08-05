import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.optim import lr_scheduler


def data(path):
    
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    transform ={
        
        'train_transforms':transforms.Compose([transforms.RandomRotation(degrees=40),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomVerticalFlip(p=0.45),
                                           transforms.RandomHorizontalFlip(p=0.35),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
        'validation_transforms' : transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])}
    df = {'train_data' : datasets.ImageFolder(path + '/train', transform = transform['train_transforms']),
               'validation_data' : datasets.ImageFolder(path + '/valid', transform = transform['validation_transforms'])
                 }
    
    dataloader = {'trainloader' : torch.utils.data.DataLoader(df['train_data'], batch_size=64),
    'validloader' : torch.utils.data.DataLoader(df['validation_data'], batch_size=32)}
    
    return dataloader['trainloader'] , dataloader['validloader'], df['train_data']


def train_model(path,arch,dropout, lr, epochs, print_every, train_loader, train_data, validation_loader, save_path, device):
    model = getattr(models, arch)(pretrained=True)
    if arch.find('vgg') ==-1: 
        n_inputs = model.classifier.in_features
    else: 
        n_inputs = model.classifier[0].in_features
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(
        nn.Linear(n_inputs, 1024),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(dropout),    
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
    model.to(device)
    
    steps = 0

    for e in range(epochs):
        model.train()
        running_loss = 0
        accuracy_train = 0
    
        for images, labels in iter(train_loader):
            steps += 1
        
            inputs, labels = Variable(images), Variable(labels)
        
            optimizer.zero_grad()
        
            
            inputs, labels = inputs.to(device), labels.to(device)
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            ps_train = torch.exp(output).data
            equality_train = (labels.data == ps_train.max(1)[1])
            accuracy_train += equality_train.type_as(torch.FloatTensor()).mean()
        
        
        
            if steps % print_every == 0:
                model.eval()
            
                accuracy = 0
                valid_loss = 0
            
                for images, labels in validation_loader:
                    with torch.no_grad():
                        inputs = Variable(images)
                        labels = Variable(labels)

                        inputs, labels = inputs.to(device), labels.to(device)

                        output = model.forward(inputs)

                        valid_loss += criterion(output, labels).item()

                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])

                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                
                print("Epoch: {}/{}.. ".format(e+1, epochs), 
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}..".format(valid_loss/len(validation_loader)),
                      "Training Accuracy: {:.3f}".format(accuracy_train/len(train_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))
            
                running_loss = 0
                model.train()
    

    state = {
    'model': model,
    'state_dict': model.state_dict(), 
    'optimizer': optimizer.state_dict(), 
    'class_to_idx': train_data.class_to_idx,
    'epochs' :epochs
    }
    
    torch.save(state, save_path)
  
    
parser = argparse.ArgumentParser(description='Train Neural Network and Save Checkpoint')

parser.add_argument('--data_path',
                    default="./flowers/")
parser.add_argument('--arch',
                    dest="arch",
                    default='vgg16')
parser.add_argument('--learning_rate',
                    dest="lr")
parser.add_argument('--dropout',
                    dest = "dropout")
parser.add_argument('--epochs',
                    dest="epoch")
parser.add_argument('--printev',
                    dest="printev")
parser.add_argument('--save_path',
                    dest="save_path",
                    default="./checkpoint.pth")
parser.add_argument('--cuda',
                    dest="cuda")

args = parser.parse_args()
data_path = args.data_path
save_path = args.save_path

arch = str(args.arch)
dropout = float(args.dropout)
lr = float(args.lr)
device = str(args.cuda)
epoch = int(args.epoch)
printev = int(args.printev)


train_loader, validation_loader, train_data = data(data_path)

train_model(data_path,arch, dropout, lr, epoch, printev, train_loader, train_data, validation_loader, save_path, device)

print('Yay! Training Completed1!!')
