import torch
import torchvision
from models import deeplab_plus
from utils import ConfMatrix  #Computes the IoU
import torch.nn as nn
from datasets import Cityscapes
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os



#device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
#device = torch.device('cuda:0')
device = torch.device('cpu')

#HYPERPARAMETERS

# Die Learningrate. Sollte klar sein
lr = 1e-4
lr_decay_speed = 0.95
weight_decay_factor = 2e-4

# Wieviele Epochen trainiert werden soll
epochs = 150

# Kuckt in Cityscapes.py. Das sind paramter die man setzen kann (Wie groß sollen die Crops aus den Bildern sein.
# wie sehr sollen die Bilder vorher schon runterskaliert werden etc.)
crop_size = None
scale_size = (513,513)
r_crop = True
r_scale = False
r_flip = False

# Trainingsbatchsize. Soviel wie die GPUs hergeben :D
batch_size = 8

# Wieviele Workerthreads im Hintergrund die Bilder vorladen sollen (das paralellisiert Torch von alleine). Einfach so lassen
num_workers = batch_size

# Falls ihr erstmal nur auf einem Subset der Bilder trainieren wollt. Argument ist eine Liste aus Integern. [i for i in range(100)]
# würde z.B. nur auf den ersten 100 Bildern trainieren
subset = None

# Naja das ist kein Hyperparameter. Ist nur wichtig für manche Funktionen die gecalled werden
num_classes = 19



# Paths and writer
# Hier werden die Pfade festgelegt, in denen die Weights gesaved werden und die Tensorboard Files hingeschrieben werden


# Hier kann man ein gespeichertes WeighFile angeben, von dem das Training fortgesetzt werden kann. (Pretraining undso)
resume_path = None

save_path = "C:/Users/mehrt/TuSoSe19/Uni/DeepML/deepmlvistas/Weights" # Hier den Pfad angeben wo die weights zwischengespeichert werden
write_path = "C:/Users/mehrt/TuSoSe19/Uni/DeepML/deepmlvistas/Runs"# Hierhin werden die Tensorboard Files geschrieben (für die Visualisierung der Loss Kurven undso)

if not os.path.exists(save_path):
    os.mkdir(save_path)

if not os.path.exists(write_path):
    os.mkdir(write_path)

writer = SummaryWriter(write_path)



#Dataset Loading
# Da müssen offensichtlich die Pfade angepasst werden! Für die anderen Parameter in Cityscapes.py reinkucken
# Die Loader sind Wrapper für die Datensets die ein gethreadetes Batch loading ermöglichen (und auch gleich mehrere Bilder zu einem großen Tensor zusammenfassen)
train_dataset = Cityscapes.Cityscape("C:\\Users\\mehrt\\Datasets\\Cityscape1026", split="fine" , subset=subset, scale_size=scale_size, crop_size=crop_size, random_crops=r_crop, random_flip=r_flip, random_scale=r_scale)
train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

val_dataset = Cityscapes.Cityscape("C:\\Users\\mehrt\\Datasets\\Cityscape1026", split="val" , subset=None, scale_size=scale_size, crop_size=crop_size, random_crops=True, random_flip=False, random_scale=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)


# Network, loading , loss, optimizer, lr_scheduler
# Wir initialisieren das Netzwerk. Dabei ist Anzahl der Outputfeatures = Anzahl Klassen (logisch)
net = deeplab_plus.DeepLabV3Plus(out_channels=num_classes).to(device)

if resume_path is not None:
    dic = torch.load(resume_path)
    net.load_state_dict(dic)

# Lossfunktion. Der Loss ist der NLLLoss (Negativ Log-Likelihood Loss). Das ist =CrossEntropy
loss = nn.NLLLoss(ignore_index=255).to(device)

# Als Optimizer nehmen wir Adam zusammen mit L2-Regularisierung (WeightDecay)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay_factor)

# Learning Rate Decay. So wie es hier steht decayed es um 5% jede Epoche
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_speed, last_epoch=-1)

# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=1, verbose=True, threshold=0.005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)




def train(start_epoch=1, save_freq=5):
    print("Saving Weights to " + save_path)
    print("Saving Tensorboard to " + write_path)
    print("Hyperparams: lr={}, batch_size={}, epochs={}".format(lr, batch_size, epochs))

    print("Starting Training")

    if start_epoch == 0:
        torch.save(net.state_dict(), save_path + "\\" + str(0) + ".wgs")
        start_acc, start_miou, start_fiou = evaluate_epoch(0)

        writer.add_scalar("epoch-val-acc", start_acc, -1)
        writer.add_scalar("epoch-val-miou", start_miou, -1)
        writer.add_scalar("epoch-val-fiou", start_fiou, -1)

    for epoch in range(start_epoch, epochs):
        print("Starting Epoch {}".format(epoch))

        average_loss, train_acc , train_miou, train_fiou = train_epoch()

        print("Completed Training on epoch {}".format(epoch))


        if epoch % save_freq == 0 or epoch == epochs-1:
            torch.save(net.state_dict(), save_path + "\\" + str(epoch) + ".wgs")

        writer.add_scalar("epoch-train-loss", average_loss, epoch)
        writer.add_scalar("epoch-train-acc", train_acc, epoch)
        writer.add_scalar("epoch-train-miou", average_loss, epoch)
        writer.add_scalar("epoch-train-fiou", train_acc, epoch)

        val_acc, val_miou, val_fiou = evaluate_epoch()

        writer.add_scalar("epoch-val-acc", val_acc, epoch)
        writer.add_scalar("epoch-val-miou", val_miou, epoch)
        writer.add_scalar("epoch-val-fiou", val_fiou, epoch)

        lr_scheduler.step()


def train_epoch():
    running_loss = 0
    num_pixels = 0
    num_correct = 0


    if device == torch.device("cpu"):
        conf_mat = ConfMatrix.ConfMatrix(num_classes, cuda=False)
    else:
        conf_mat = ConfMatrix.ConfMatrix(num_classes, cuda=True)


    net.train()

    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        img = batch[0].to(device)
        mask = batch[1].to(device)

        out = net(img)
        preds = torch.argmax(out, dim=1)
        l = loss(out, mask)
        running_loss = running_loss + l.item()
        l.backward()
        optimizer.step()

        num_pixels = num_pixels + (mask.size(0) * mask.size(1) * mask.size(2)) - torch.sum(mask == 255).item()

        pm = preds == mask
        un255 = mask != 255
        num_correct += torch.sum(pm & un255).item()
        conf_mat.addPred(mask, preds)

    return running_loss / len(train_dataset), num_correct / (num_pixels+1), conf_mat.getMIoU(), conf_mat.getfIoU()


def evaluate_epoch():
    num_pixels = 0
    num_correct = 0

    if device == torch.device("cpu"):
        conf_mat = ConfMatrix.ConfMatrix(num_classes, cuda=False)
    else:
        conf_mat = ConfMatrix.ConfMatrix(num_classes, cuda=True)


    net.eval()

    for i, batch in enumerate(tqdm(val_loader)):
        img = batch[0].to(device)
        mask = batch[1].to(device)

        out = net(img)

        preds = torch.argmax(out, dim=1)

        num_pixels = num_pixels + (mask.size(0) * mask.size(1) * mask.size(2)) - torch.sum(mask == 255).item()
        pm = preds == mask
        un255 = mask != 255
        num_correct += torch.sum(pm & un255).item()

        conf_mat.addPred(mask, preds)

    return num_correct / num_pixels, conf_mat.getMIoU(), conf_mat.getfIoU()


if __name__ == "__main__":

    #Training beginnt bei Epoche 1 (für Dateinamen wichtig) und es wird alle 5 Epochen gespeichert
    train(1, save_freq=5)


    #So kann man z.B. die Predictions visualisieren. In diesem Fall von Bild nummer 0.
    pred = torch.argmax(net(train_dataset[0][0].unsqueeze(0)),1).squeeze(0)
    mask = train_dataset[0][1]
    img = train_dataset.load_original(0)
    Cityscapes.compareImgPil(img, mask, pred).show()

