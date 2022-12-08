import torch


def validation(val_loader, model, loss_fn, epoch, writer, device, tag='Validation'):

    # 开始测试
    model.eval()
    total_acc = 0
    total_loss = 0

    with torch.no_grad():

        for data in val_loader:
            imgs, targets = data
            imgs = imgs.cuda(device)

            targets = targets.cuda(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            # measure accuracy and record loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_acc = total_acc + accuracy
            total_loss = total_loss + loss.item()
        top1 = total_acc/(len(val_loader)*val_loader.batch_size)
        print('{} : in epoch {}, the accuracy is : {}'.format(tag, epoch, top1))
    writer.add_scalar('valid_loss', total_loss, epoch)
    return top1
