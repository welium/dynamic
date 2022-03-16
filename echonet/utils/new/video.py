"""Functions for training and running EF prediction."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

from .models import r2plus1d_18_unc

echonet = __import__(__name__.split('.')[0])


@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default="EF")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", type=int, default=45)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--drop_rate", type=float, default=0.2)
@click.option("--val_samp", type=int, default=1)

def worker_init_fn(worker_id):                            
    # print("worker id is", torch.utils.data.get_worker_info().id)
    # https://discuss.pytorch.org/t/in-what-order-do-dataloader-workers-do-their-job/88288/2
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def run(
    data_dir=None,
    output=None,
    task="EF",

    pretrained=True,
    weights=None,

    run_test=False,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
    drop_rate = 0.2,
    val_samp=1
):
    """Trains/tests EF prediction model.

    \b
    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/video/<model_name>_<pretrained/random>/.
        task (str, optional): Name of task to predict. Options are the headers
            of FileList.csv. Defaults to ``EF''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training.
            Defaults to 45.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 1e-4.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to 15.
        frames (int, optional): Number of frames to use in clip
            Defaults to 32.
        period (int, optional): Sampling period for frames
            Defaults to 2.
        n_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
        drop_rate(float, optional): Drop Rate of the models. Defaults to 0.2.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    num_of_gpus = torch.cuda.device_count()
    print("Available gpus: " + str(num_of_gpus))

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_name = "r2plus1d_unc"
    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using " + str(device) + " for training")

    # Set up model
    model = r2plus1d_18_unc(num_classes=1, pretrained=pretrained, drop_rate=drop_rate)
    model.to(device)
    print("Bias of fc_1 in model is :{:.2f}".format(model.fc_1.bias.data[0]))

    model_1 = r2plus1d_18_unc(num_classes=1, pretrained=pretrained, drop_rate=drop_rate)
    model_1.to(device)
    print("Bias of fc_1 in model1 is :{:.2f}".format(model_1.fc_1.bias.data[0]))
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
        model_1 = torch.nn.DataParallel(model_1)
   
    if weights is not None:
        checkpoint = torch.load(weights)
        if checkpoint.get('state_dict'):
            model.load_state_dict(checkpoint['state_dict'])
        elif checkpoint.get('state_dict_0'):
            model.load_state_dict(checkpoint['state_dict_0'])
        else:
            assert 1==2, "state dict not found"

    # Set up optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("optim", optim)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    optim_1 = torch.optim.Adam(model_1.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optim_1, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period
              }

    # Set up datasets and dataloaders
    multiplier = 8 # Labelled dataset is 1/8 of unlabelled dataset
    dataset = {}
    dataset_train = {}
    dataset_train["labelled"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12, multiplier = multiplier)
    dataset_train["unlabelled"] = echonet.datasets.Echo(root=data_dir, split="train_unlabelled", **kwargs, pad=12)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)
    dataset["train"] = dataset_train
    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])

            model_1.load_state_dict(checkpoint['state_dict_1'])
            optim_1.load_state_dict(checkpoint['opt_dict_1'])
            scheduler_1.load_state_dict(checkpoint['scheduler_dict_1'])

            np_randstate = checkpoint['np_randstate']
            torch_randstate = checkpoint['torch_randstate']

            np.random.set_state(np_randstate)
            torch.set_rng_state(torch_randstate)

            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                if phase == 'train':
                    labelled_loader = torch.utils.data.DataLoader(
                        ds['labelled'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)
                    unlablled_loader = torch.utils.data.DataLoader(
                        ds['unlabelled'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)
                    print("Running epoch: {}, split: {}".format(epoch, phase))
                    #TODO: rewrite this part after change run_epoch
                    # loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, phase == "train", optim, device)
                    yhat_0, yhat_1, loss_tr, loss_reg_0, cps = 0,0,0,0,0
                    r2_value_0 = sklearn.metrics.r2_score(y, yhat_0)
                    r2_value_1 = sklearn.metrics.r2_score(y, yhat_1)

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                loss_tr,
                                                                r2_value_0,
                                                                r2_value_1,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                loss_reg_0,
                                                                cps))
                    f.flush()
                elif phase == 'val':
                    np_randstate = np.random.get_state()
                    torch_randstate = torch.get_rng_state()
                    r2_track = []
                    loss_track = []

                    for val_samp_itr in range(val_samp):
                        print("running validation batch for seed =", val_samp_itr)

                        np.random.seed(val_samp_itr)
                        torch.manual_seed(val_samp_itr)
    
                        ds = dataset[phase]
                        dataloader = torch.utils.data.DataLoader(
                            ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))                        
                        # TODO: write it after finish run_epoch_val
                        # , yhat, y, var_hat, var_e, var_a, mean_0_ls, var_0_ls = run_epoch_val(model = model, dataloader = dataloader, train = False, optim = None, device = device, save_all=False, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq)
                        loss_valit = 0
                        r2_track.append(sklearn.metrics.r2_score(y, yhat))
                        loss_track.append(loss_valit)

                    r2_value = np.average(np.array(r2_track))
                    loss = np.average(np.array(loss_track))

                    f.write("{},{},{},{},{},{},{},{},{},{},{}".format(epoch,
                                                                phase,
                                                                loss,
                                                                r2_value,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                0,
                                                                0))
                    for trck_write in range(len(r2_track)):
                        f.write(",{}".format(r2_track[trck_write]))
                    for trck_write in range(len(loss_track)):
                        f.write(",{}".format(loss_track[trck_write]))
                    for val_samp_itr in range(val_samp):
                        print("running validation batch for seed =", val_samp_itr)

                        np.random.seed(val_samp_itr)
                        torch.manual_seed(val_samp_itr)
    
                        ds = dataset[phase]
                        dataloader = torch.utils.data.DataLoader(
                            ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))                        
                        # TODO: write it after finish run_epoch_val
                        # , yhat, y, var_hat, var_e, var_a, mean_0_ls, var_0_ls = run_epoch_val(model = model, dataloader = dataloader, train = False, optim = None, device = device, save_all=False, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq)
                        loss_valit_1 = 0
                        r2_track.append(sklearn.metrics.r2_score(y, yhat))
                        loss_track.append(loss_valit)

                    r2_value = np.average(np.array(r2_track))
                    loss = np.average(np.array(loss_track))

                    f.write("{},{},{},{},{},{},{},{},{},{},{}".format(epoch,
                                                                phase,
                                                                loss,
                                                                r2_value,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size,
                                                                0,
                                                                0))
                    for trck_write in range(len(r2_track)):
                        f.write(",{}".format(r2_track[trck_write]))
                    for trck_write in range(len(loss_track)):
                        f.write(",{}".format(loss_track[trck_write]))

            scheduler.step()
            scheduler_1.step()

            if loss_valit_1 < loss_valit:
                best_model_loss = loss_valit_1
                best_weights = model_1.state_dict()
            else:
                best_model_loss = loss_valit
                best_weights = model.state_dict()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'best_weights': best_weights,
                'loss': loss,
                'r2': r2_value,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'opt_dict_1': optim_1.state_dict(),
                'scheduler_dict_1': scheduler_1.state_dict(),
                'np_randstate': np.random.get_state(),
                'torch_randstate': torch.get_rng_state()                
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if best_model_loss < bestLoss:
                print("saved best because {} < {}".format(best_model_loss, bestLoss))
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = best_model_loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}, R2 {}\n".format(checkpoint["best_model_loss"], checkpoint["epoch"], checkpoint["r2"]))
            f.flush()

        if run_test:
            for split in ["val", "test"]:
                print("Without test-time augmentation, split: {}".format(split))
                # Performance without test-time augmentation
                dataloader = torch.utils.data.DataLoader(
                    echonet.datasets.Echo(root=data_dir, split=split, **kwargs),
                    batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device)
                f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                f.flush()
                print("With test-time augmentation, split: {}".format(split))
                # Performance with test-time augmentation
                ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, yhat, y = echonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=batch_size)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, np.array(list(map(lambda x: x.mean(), yhat))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Write full performance to file
                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                echonet.utils.latexify()
                yhat = np.array(list(map(lambda x: x.mean(), yhat)))

                # Plot actual and predicted EF
                fig = plt.figure(figsize=(3, 3))
                lower = min(y.min(), yhat.min())
                upper = max(y.max(), yhat.max())
                plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y > thresh, yhat)
                    print(thresh, sklearn.metrics.roc_auc_score(y > thresh, yhat))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)


def run_epoch(model, model_1, dataloader_lb, dataloader_ul, train, optim, optim_1, device, save_all=False, 
                w_cps=1, y_mean=43.3, y_std=36, epiunc = True, samp_ssl = 3, w_aliv=1):
    """Run one epoch of training for segmentation."""
    model.train(train)
    model_1.train(train)

    total = 0  # total training loss
    total_reg = 0 
    total_reg_1 = 0

    total_cps = 0
    total_cps_0 = 0
    total_cps_1 = 0

    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat_0 = []
    yhat_1 = []
    y = []

    mean2s_0_stack_ls = []
    mean2s_1_stack_ls = []
    var1s_0_stack_ls = []
    var1s_1_stack_ls = []

    start_frame_record = []

    total_iteration = min(len(dataloader_lb), len(dataloader_ul))
    torch.set_grad_enabled(train)

    lb_iterator = iter(dataloader_lb)
    ul_iterator = iter(dataloader_ul)
    
    for iteration in range(total_iteration):
        (X_ul, _) = ul_iterator.next()
        X_ul.to(device)

        if train:
            

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])

                if save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(outputs.view(-1), outcome)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    if not save_all:
        yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / n, yhat, y
