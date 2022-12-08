import time


def train(train_loader, model, loss_fn, optim, epoch, writer, device):

    # 训练模式
    model.train()
    end = time.time()
    total_acc = 0
    total_loss = 0
    start_time = time.time()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.cuda(device)

        targets = targets.cuda(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 计算损失和准确率
        accuracy = (outputs.argmax(1) == targets).sum()
        total_acc = total_acc + accuracy
        total_loss = total_loss + loss.item()

        # 优化模型更新梯度
        optim.zero_grad()
        loss.backward()
        optim.step()

    # 记录训练一个epoch的时间
    end_time = time.time()
    print('in Epoch {}:\t'
          'loss: {}\t'
          'accuracy: {}\t'
          'time : {}'.format(epoch, total_loss, total_acc/(len(train_loader)*train_loader.batch_size),
                             end_time-start_time))
    writer.add_scalar('train_loss', total_loss, epoch)
    return


