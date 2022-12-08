import torch
import torchvision
from PIL import Image
from model import resnet50


def infer(img, device):
    # 推理
    classes_num = 214  # 分类任务的类别数目
    model = resnet50.Clean(classes_num, weights=False)
    model = model.cuda(device)
    ckp = torch.load('../model_best_checkpoint_resnet50.pth.tar')
    model.load_state_dict(ckp['state_dict'])
    model.eval()
    with torch.no_grad():
        img = img.cuda(device)
        predict = model(img).argmax(1)
    return predict


def padding_black(img):

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


def get_label_dict(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
        labels = list(map(lambda x: x.strip().split('\t'), labels))
        list(map(lambda x: x.pop(), labels))
        list(map(lambda x: x.pop(), labels))
        return labels


if __name__ == '__main__':

    lab_dict = get_label_dict('../label_dir.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = 224
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_path = '../垃圾图片库/其他垃圾_唱片/img_唱片_25.jpeg'
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = padding_black(img)
    img = trans(img)
    img = torch.reshape(img, (1, 3, size, size))
    res = infer(img, device)
    print(lab_dict[res])






