# #!/usr/bin/python
# -*- coding: utf-8 -*-
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GarbageDataset(Dataset):

    def __init__(self, txt_path, train_flag=True) -> None:
        super().__init__()
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.trans_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
        self.trans_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x: x.strip().split('\t'), imgs_info))
        return imgs_info

    def padding_black(self, img):

        w, h = img.size

        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

        size_fg = img_fg.size
        size_bg = 224

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.padding_black(img)
        if self.train_flag:
            img = self.trans_train(img)
        else:
            img = self.trans_test(img)

        return img, int(label)

    def __len__(self):
        return len(self.imgs_info)


if __name__ == '__main__':

    train_dataset = GarbageDataset("../train.txt", True)
    print("total samples nums：", len(train_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label)
