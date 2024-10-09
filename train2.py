import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import os
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from my_dataset import MyDataSetKFold
from model import mobile_vit_small as create_model
from k_utils import read_split_data, train_one_epoch, evaluate

class MyDataSetKFold(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    images_path, images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Splitting the dataset into K folds
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    fold_datasets = []

    for train_index, val_index in kf.split(images_path):
        train_images_path = [images_path[i] for i in train_index]
        train_images_label = [images_label[i] for i in train_index]
        val_images_path = [images_path[i] for i in val_index]
        val_images_label = [images_label[i] for i in val_index]

        # Instantiate training dataset
        train_dataset = MyDataSetKFold(images_path=train_images_path,
                                       images_class=train_images_label,
                                       transform=data_transform["train"])

        # Instantiate validation dataset
        val_dataset = MyDataSetKFold(images_path=val_images_path,
                                     images_class=val_images_label,
                                     transform=data_transform["val"])

        fold_datasets.append((train_dataset, val_dataset))

    best_acc = 0.0
    for fold, (train_dataset, val_dataset) in enumerate(fold_datasets):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

        model = create_model(num_classes=args.num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1E-2)

        for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            # validate
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

            if val_acc >= best_acc:
                best_acc = val_acc
            torch.save(model.state_dict(), f"./weights/best_model_fold{fold+1}.pth".format(fold+1))

            torch.save(model.state_dict(), "./weights/latest_model_fold{}.pth".format(fold+1))

    tb_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--data_path', type=str, default=r"D:\pycharmproject\YMQ\san")
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
