import argparse
from torch.utils.data import DataLoader
from sklearn import metrics
from learner import Learner
from loss import *
from dataset import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='D:/137/dataset/VAD/features/UCF_and_Shanghai/')
    parser.add_argument('--batch-size', default=60)
    parser.add_argument('--modality', default='rgb', choices=['rgb', 'flow', 'two-stream'])
    args = parser.parse_args()
    return args


def train(model, optimizer, scheduler, criterion, normal_train_loader, anomaly_train_loader):
    model.train()
    train_loss = 0
    
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        if anomaly_inputs.shape[0] != normal_inputs.shape[0]:
            if np.argmax([anomaly_inputs.shape[0], normal_inputs.shape[0]]) == 0:
                anomaly_inputs = anomaly_inputs[:normal_inputs.shape[0]]
            else:
                normal_inputs = normal_inputs[:anomaly_inputs.shape[0]]
        
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    print(f'loss = {train_loss/min(len(normal_train_loader), len(anomaly_train_loader))}')
    scheduler.step()


def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        score_list_all = np.zeros(0)
        gt_list_all = np.zeros(0)
        
        for i, data in enumerate(test_loader):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, torch.div(frames[0], 16, rounding_mode='floor'), 33))
            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]

            if gts == -1:
                gt_list = np.zeros(frames[0])
            else:
                gt_list = np.zeros(frames[0])
                for k in range(len(gts)//2):
                    s = gts[k*2]
                    e = min(gts[k*2+1], frames)
                    gt_list[s-1:e] = 1
            
            score_list_all = np.concatenate((score_list_all, score_list), axis=0)
            gt_list_all = np.concatenate((gt_list_all, gt_list), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print(f'auc = {auc}')


if __name__ == '__main__':
    args = parse_args()

    normal_train_dataset = Normal_Dataset(is_train=1, data_root=args.data_root, modality=args.modality)
    anomaly_train_dataset = Anomaly_Dataset(is_train=1, data_root=args.data_root, modality=args.modality)
    test_dataset = Test_Dataset(data_root=args.data_root, modality=args.modality)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=args.batch_size, shuffle=True)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=args.batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    print(f'batch size: {args.batch_size}\n'
          f'modality: {args.modality}\n'
          f'normal train dataset length: {len(normal_train_loader)}\n'
          f'anomaly train dataset length: {len(anomaly_train_loader)}\n'
          f'test dataset length: {len(test_loader)}\n'
          f'batch nums in one epoch: {min(len(normal_train_loader), len(anomaly_train_loader))}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.modality == 'two-stream':
        input_dim = 2048
    elif args.modality == 'rgb' or 'flow':
        input_dim = 1024
    
    model = Learner(input_dim=input_dim, drop_p=0.6).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr= 0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    criterion = MIL

    print(f'\nEpoch: 0')
    test(model, test_loader)
    for epoch in range(75):
        print(f'\nEpoch: {epoch + 1}')
        train(model, optimizer, scheduler, criterion, normal_train_loader, anomaly_train_loader)
        test(model, test_loader)
