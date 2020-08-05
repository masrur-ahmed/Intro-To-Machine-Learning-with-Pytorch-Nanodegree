import argparse
import torch, os, json
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

def predict(path, checkpoint, topk=5, labels=None, device = 'cpu'):
    model = 0
    if os.path.isfile(checkpoint):
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        model = model.to(device)
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        
    model.idx_to_class = inv_map = {v: k for k, v in model.class_to_idx.items()}
    img = Image.open(path)

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    inputs = trans(img)
        
    model_p = model.eval()
    
    inputs = Variable(inputs.unsqueeze(0))

    inputs = inputs.to(device)
    
    output = model_p(inputs)
    x = torch.exp(output).data
    
    top = x.topk(topk)
    idx2class = model.idx_to_class
    probs = top[0].tolist()[0]
    classes = [idx2class[i] for i in top[1].tolist()[0]]

    if labels is not None:
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)

        class_names = [cat_to_name[i] for i in classes]
    print("Labels = {}".format(classes))
    print("Probabilities = {}".format(probs))
    print("Classes = {}".format(class_names))




parser = argparse.ArgumentParser(description='Predict Neural Network from Saved CheckPoint')

parser.add_argument('--path',
                   dest = "path")
parser.add_argument('--checkpoint',
                   dest = "checkpoint")
parser.add_argument('--topk',
                    dest="topk")
parser.add_argument('--labels',
                    dest="labels")
parser.add_argument('--gpu',
                    dest="gpu")

arg = parser.parse_args()


path_img = str(arg.path)
device = str(arg.gpu)
path = str(arg.checkpoint)
topk = int(arg.topk)
labels = str(arg.labels)
arguments = parser.parse_args()

predict(path_img, path, topk=topk, labels=labels, device = 'cuda')
print('Done!')