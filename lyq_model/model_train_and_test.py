from esmm_din_ppnet import *
from model_utils import dayid
from random import random


def train(model, train_loader, dayid, device='cpu', save_dir='./', resume=''):
    # optim and loss
    optimizer = Adam(model.parameters(), lr=1e-3)
    focal_loss = FocalLoss()
    mse_loss = MSELoss()

    # load
    if resume != '':
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Day{checkpoint['day']}'s model has been loaded!")

    model.to(device)
    model.train()

    for i, batch in enumerate(train_loader):
        sample = {k: v.to(device) for k, v in batch.items()}
        click_label, ptr_label = sample['click'], sample['ptr']
        # model
        click_logit, play_logit = model(sample)
        # loss
        click_loss = focal_loss(click_logit, click_label)
        ptr = F.sigmoid(click_logit) * F.sigmoid(play_logit)
        play_loss = mse_loss(ptr, ptr_label)
        loss = click_loss + play_loss

        # propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(f"batch: {i}, loss: {loss}")

    # save
    os.makedirs(save_dir, exist_ok=True)
    saved_path = save_dir + f"day{dayid}.pt"
    torch.save({
        'day': dayid,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, saved_path)
    print("--" * 40)
    print(f"Day{dayid} finished! Checkpoint has been saved in {saved_path}")
    print("--" * 40)


def test(model, test_loader, dayid, sample=1.0, device='cpu', resume=''):
    if resume != '':
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    all_click_preds = []
    all_click_labels = []
    all_ptr_preds = []
    all_ptr_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if random() > sample:
                continue
            sample = {k: v.to(device) for k, v in batch.items()}
            click_label = sample['click'].to(device)
            ptr_label = sample['ptr'].to(device)
            click_logit, ptr_logit = model(sample)

            all_click_preds.append(torch.sigmoid(click_logit).cpu())
            all_click_labels.append(click_label.cpu())

            all_ptr_preds.append(ptr_logit.cpu())
            all_ptr_labels.append(ptr_label.cpu())


    click_probs = torch.cat(all_click_preds).numpy()
    click_labels = torch.cat(all_click_labels).numpy()
    ptr_preds = torch.cat(all_ptr_preds)
    ptr_labels = torch.cat(all_ptr_labels)

    auc = roc_auc_score(click_labels, click_probs)
    mse = F.mse_loss(ptr_preds, ptr_labels).item()

    print("--"*20 + f"Evaluating on day{dayid}" + "--"*20)
    print(f"click AUC: {auc:.6f}, ptr MSE: {mse:.6f}")
    print("--"*40)


if __name__ == '__main__':
    # model selected
    model = "esmm_din"      # 选model_name
    # dir selected
    data_dir = 'E:/芒果台比赛/train_data'        # 训练数据地址
    save_dir = f'E:/芒果台比赛/{model}_checkpoint'       # 检查点保存地址

    # dataloader settings
    device = 'cpu'      # 调device
    batch_size = 128
    num_workers = 16       # 调num_workers
    prefetch_factor = 4       # 调prefetch_factor
    persistent_workers = True

    # initiate model
    if model == "esmm_din":
        model = ESMM_DIN()
    elif model == "esmm_din_ppnet":
        model = ESMM_DIN_PPNET()
    else:
        raise ValueError("what is this?")

    # initiate dataset and dataloader
    dayid = dayid(2)
    dataset = MangoDataset(data_dir=data_dir, dayid=dayid)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=False,
                            prefetch_factor=prefetch_factor,
                            persistent_workers=persistent_workers)

    # train and eval
    for i in range(2, 30+1):
        train(model, dataloader, dayid, device=device, save_dir=save_dir)

        if i == 30:
            break

        next_dayid = dayid(i+1)
        dataset = MangoDataset(data_dir=data_dir, dayid=next_dayid)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=False,
                                prefetch_factor=prefetch_factor,
                                persistent_workers=persistent_workers)
        test(model, dataloader, next_dayid, sample=0.05)


