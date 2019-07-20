import  torch
from torchvision import datasets,transforms
from torch.utils.data  import  DataLoader
from torch import optim,nn
from lenet5_dows import  Lenet5
from restnet_dows import rest18, resblk

def main():
    batch_size = 32
    train_data = datasets.CIFAR10('cifar',train=True,transform=transforms.Compose([
        transforms.Resize(32,32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),download=True)
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)

    test_data = datasets.CIFAR10('cifar',train=False,transform=transforms.Compose([
        transforms.Resize(32,32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),download=True)

    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

    #x, label = iter(train_loader).next()
    # x, label = iter(train_loader).next()
    # print('x:', x.shape, 'label', label.shape)
    lr = 0.001
    #model = Lenet5()
    model = rest18()
    print(model)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    citeron = nn.CrossEntropyLoss()
    for eporch in range(10000):
        for batch_idx, (x, label) in enumerate(train_loader):
            #[b,3,32,32]
            logits = model(x)
            loss = citeron(logits,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 700 == 0:
                print('eporch:',eporch,'batch_idx:',batch_idx,'loss:',loss.item())

        correct_num = 0
        total_num = 0
        loss = 0
        for x, label in test_loader:
            out = model(x)
            loss += citeron(out, label)
            pred = out.argmax(dim=1)

            correct_num += pred.eq(label).float().sum().item()
        total_num = len(test_loader.dataset)
        acc = correct_num / total_num
        loss = loss/ total_num
        print('acc:',acc,'loss_ave:',loss.item())



if __name__ == '__main__':
    main()
