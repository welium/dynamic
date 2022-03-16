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
@click.option("--w_cps", type=int, default=1)
@click.option("--epiunc/--cmbunc", default=False)
@click.option("--samp_ssl", type=int, default=3)
@click.option("--w_aliv", type=int, default=1)
@click.option("--samp_fq", type=int, default=1)

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
    w_cps = 1,
    epiunc = False,
    samp_ssl = 3,
    w_aliv = 1,

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
    val_samp=1,
    samp_fq = 10
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
    num_of_gpus = torch.cuda.device_count()
    print("Available gpus: " + str(num_of_gpus))

    y_mean = 43.3 #calculated elsewhere
    y_std = 36 #calculated elsewhere

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
                    loss_tr, loss_reg_0, loss_reg_1, cps, cps_l, cps_s, yhat_0, yhat_1, y, mean_0_ls, mean_1_ls, var_0_ls, var_1_ls = run_epoch(model, 
                                                                                                                                model_1, 
                                                                                                                                labelled_loader, 
                                                                                                                                unlablled_loader, 
                                                                                                                                phase == "train", 
                                                                                                                                optim, 
                                                                                                                                optim_1, 
                                                                                                                                device, 
                                                                                                                                w_cps = w_cps, 
                                                                                                                                y_mean = y_mean, 
                                                                                                                                y_std = y_std, 
                                                                                                                                epiunc = epiunc, 
                                                                                                                                samp_ssl = samp_ssl, 
                                                                                                                                w_aliv = w_aliv
                    )

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

                    with open(os.path.join(output, "train_pred_{}.csv".format(epoch)), "w") as f_trnpred:
                        for clmn in range(mean_0_ls.shape[1]):
                            f_trnpred.write("m_0_{},".format(clmn))
                        for clmn in range(mean_1_ls.shape[1]):
                            f_trnpred.write("m_1_{},".format(clmn))
                        for clmn in range(var_0_ls.shape[1]):
                            f_trnpred.write("v_0_{},".format(clmn))
                        for clmn in range(var_1_ls.shape[1]):
                            f_trnpred.write("v_1_{},".format(clmn))
                        f_trnpred.write("\n".format(clmn))
                        
                        for rw in range(mean_0_ls.shape[0]):
                            for clmn in range(mean_0_ls.shape[1]):
                                f_trnpred.write("{},".format(mean_0_ls[rw, clmn]))
                            for clmn in range(mean_1_ls.shape[1]):
                                f_trnpred.write("{},".format(mean_1_ls[rw, clmn]))
                            for clmn in range(var_0_ls.shape[1]):
                                f_trnpred.write("{},".format(var_0_ls[rw, clmn]))
                            for clmn in range(var_1_ls.shape[1]):
                                f_trnpred.write("{},".format(var_1_ls[rw, clmn]))
                            f_trnpred.write("\n".format(clmn))

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
                        loss_valit, yhat, y, var_hat, var_e, var_a, mean_1_ls, var_1_ls = run_epoch_val(
                            model = model, dataloader = dataloader, train = False, optim = None, device = device, 
                            save_all=False, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq
                            )          
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
                        loss_valit_1, yhat, y, var_hat, var_e, var_a, mean_1_ls, var_1_ls = run_epoch_val(
                            model = model_1, dataloader = dataloader, train = False, optim = None, device = device, 
                            save_all=False, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq
                            )     
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
                                                                1,
                                                                1))
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
                for seed_itr in range(1):
                    np.random.seed(seed_itr)
                    torch.manual_seed(seed_itr)                
                    dataloader = torch.utils.data.DataLoader(
                        echonet.datasets.Echo(root=data_dir, split=split, **kwargs),
                        batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"))

                    temp = run_epoch_val(model = model, dataloader = dataloader, train = False, optim = None, device = device, save_all=False, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq)
                    y = temp[1]
                    yhat = temp[2]
                    f.write("Seed is {}".format(seed_itr))
                    f.write("{} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                    f.write("{} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                    f.write("{} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                    f.flush()


                    print("With test-time augmentation, split: {}".format(split))
                    # Performance with test-time augmentation
                    ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips="all")
                    dataloader = torch.utils.data.DataLoader(
                        ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=(device.type == "cuda"))
                    # num_workers needs to be 0 or some weird bugs on multithreading would happen in this stage
                    temp = run_epoch_val(model = model, dataloader = dataloader, train = False, optim = None, device = device, save_all=False, block_size=None, y_mean = y_mean, y_std = y_std, samp_fq = samp_fq)
                    y = temp[1]
                    yhat = temp[2]
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

    means_0_stack_ls = []
    means_1_stack_ls = []
    vars_0_stack_ls = []
    vars_1_stack_ls = []

    start_frame_record = []

    total_iteration = min(len(dataloader_lb), len(dataloader_ul))
    torch.set_grad_enabled(train)

    lb_iterator = iter(dataloader_lb)
    ul_iterator = iter(dataloader_ul)
    
    for iteration in range(total_iteration):
        # Unlabeled Forward pass
        (X_ul, _) = ul_iterator.next()
        X_ul = X_ul.to(device)
        if train:
            output_ul_pred_0, var_ul_pred_0 = model(X_ul)
            output_ul_pred_1, var_ul_pred_1 = model_1(X_ul)
        else:
            with torch.no_grad():
                output_ul_pred_0, var_ul_pred_0 = model(X_ul)
                output_ul_pred_1, var_ul_pred_1 = model_1(X_ul)
        
        means_0 = []
        vars_0 = []

        means_1 = []
        vars_1 = []

        X_ulb_in = X_ul

        # Repeat 3 times to get random dropout result
        with torch.no_grad():
            for samp_ssl_itr in range(samp_ssl):
                mean_raw_0, var_raw_0 = model(X_ulb_in)
                mean_0 = mean_raw_0.view(-1)
                var_0 = var_raw_0.view(-1)
                means_0.append(mean_0)
                vars_0.append(var_0)

                mean_raw_1, var_raw_1 = model_1(X_ulb_in)
                mean_1 = mean_raw_1.view(-1)
                var_1 = var_raw_1.view(-1)
                means_1.append(mean_1)
                vars_1.append(var_1)

        # Calculation for pseuodo label based on the dropout result
        means_0_stack = torch.stack(means_0, dim=1).to("cpu").detach().numpy()
        means_0_stack_ls.append(means_0_stack)
        vars_0_stack = torch.stack(vars_0, dim=1).to("cpu").detach().numpy()
        vars_0_stack_ls.append(vars_0_stack)

        means_0_mean = torch.stack(means_0, dim=0).mean(dim=0)
        vars_0_mean = torch.stack(vars_0, dim=0).mean(dim=0)

        means_1_stack = torch.stack(means_1, dim=1).to("cpu").detach().numpy()
        means_1_stack_ls.append(means_1_stack)
        vars_1_stack = torch.stack(vars_1, dim=1).to("cpu").detach().numpy()
        vars_1_stack_ls.append(vars_1_stack)

        means_1_mean = torch.stack(means_1, dim=0).mean(dim=0)
        vars_1_mean = torch.stack(vars_1, dim=0).mean(dim=0)

        pseudolabel_0 = means_0_mean
        pseudolabel_1 = means_1_mean

        pseudolabel_mean = pseudolabel_0 * 0.5 + pseudolabel_1 * 0.5
        var_mean = vars_0_mean * 0.5 + vars_1_mean * 0.5

        # Loss Based on pseudolabel: want low variance and close to mean prediction
        loss_mse_cps_0 = ((output_ul_pred_0.view(-1) - pseudolabel_mean)**2)
        loss_mse_cps_1 = ((output_ul_pred_1.view(-1) - pseudolabel_mean)**2)
        loss_cmb_cps_0 = 0.5 * (torch.mul(torch.exp(-var_mean), loss_mse_cps_0) + var_mean)
        loss_cmb_cps_1 = 0.5 * (torch.mul(torch.exp(-var_mean), loss_mse_cps_1) + var_mean)
        loss_reg_cps0 = loss_cmb_cps_0.mean()
        loss_reg_cps1 = loss_cmb_cps_1.mean()
        var_loss_ulb_0 = ((var_ul_pred_0.view(-1) - var_mean)**2).mean()
        var_loss_ulb_1 = ((var_ul_pred_1.view(-1) - var_mean)**2).mean()
        loss_reg_cps = (loss_reg_cps0 + loss_reg_cps1) + w_aliv * (var_loss_ulb_0 + var_loss_ulb_1)

        # Labelled Forward Pass
        (X, outcome) = lb_iterator.next()
        y.append(outcome.detach().cpu().numpy())
        X = X.to(device)
        outcome = outcome.to(device)
        s1 += outcome.sum()
        s2 += (outcome ** 2).sum()

        if train:
            pred_lb_0, var_lb_0 = model(X)
            pred_lb_1, var_lb_1 = model_1(X)                    
        else:
            with torch.no_grad():                    
                pred_lb_0, var_lb_0 = model(X)
                pred_lb_1, var_lb_1 = model_1(X)     

        # Handle Loss for labelled, model 0
        pred_lb_0 = pred_lb_0.view(-1)
        var_lb_0 = var_lb_0.view(-1)
        loss_mse_0 = (pred_lb_0 - (outcome - y_mean) / y_std) ** 2
        loss1_0 = torch.mul(torch.exp(-var_lb_0), loss_mse_0)
        loss2_0 = var_lb_0
        loss_0 = .5 * (loss1_0 + loss2_0)
        if epiunc:        
            loss_reg_0 = loss_mse_0.mean()
        else:
            loss_reg_0 = loss_0.mean()
        yhat_0.append(pred_lb_0.view(-1).to("cpu").detach().numpy() * y_std + y_mean)

        # Handle Loss for labelled, model 1
        pred_lb_1 = pred_lb_1.view(-1)
        var_lb_1 = var_lb_1.view(-1)
        loss_mse_1 = (pred_lb_1 - (outcome - y_mean) / y_std) ** 2
        loss1_1 = torch.mul(torch.exp(-var_lb_1), loss_mse_1)
        loss2_1 = var_lb_1
        loss_1 = .5 * (loss1_1 + loss2_1)
        if epiunc:        
            loss_reg_1 = loss_mse_1.mean()
        else:
            loss_reg_1 = loss_1.mean()
        yhat_1.append(pred_lb_1.view(-1).to("cpu").detach().numpy() * y_std + y_mean)

        loss_reg = (loss_reg_0 + loss_reg_1)

        loss = loss_reg + w_cps * loss_reg_cps + w_aliv((var_1 - var_0) ** 2).mean()

        if train:
            optim.zero_grad()
            optim_1.zero_grad()
            loss.backward()
            optim.step()
            optim_1.step()

        total += loss.item() * outcome.size(0)
        total_reg += loss_reg_0.item() * outcome.size(0)
        total_reg_1 += loss_reg_1.item() * outcome.size(0)

        total_cps += loss_reg_cps.item() * outcome.size(0)
        total_cps_0 += loss_reg_cps0.item() * outcome.size(0)
        total_cps_1 += loss_reg_cps1.item() * outcome.size(0)
        n += outcome.size(0)

        if iteration % 10 == 0:
            # break
            print("phase {} itr {}/{}: ls {:.2f}({:.2f}) rg0 {:.4f} ({:.2f}) rg1 {:.4f} ({:.2f}) cps {:.4f} ({:.2f}) cps0 {:.4f} ({:.2f}) cps1 {:.4f} ({:.2f})".format(train,
                iteration, total_iteration, 
                total / n, loss.item(), 
                total_reg/n, loss_reg_0.item(), 
                total_reg_1/n, loss_reg_1.item(), 
                total_cps/n, loss_reg_cps.item(),
                total_cps_0/n, loss_reg_cps0.item(),
                total_cps_1/n, loss_reg_cps1.item()), flush = True)

    if not save_all:
        yhat_0 = np.concatenate(yhat_0)
        yhat_1 = np.concatenate(yhat_1)
        if not train:
            start_frame_record = np.concatenate(start_frame_record)
        # vidpath_record = np.concatenate(vidpath_record)

    y = np.concatenate(y)

    means_0_stack_ls = np.concatenate(means_0_stack_ls)
    means_1_stack_ls = np.concatenate(means_1_stack_ls)
    vars_0_stack_ls = np.concatenate(vars_0_stack_ls)
    vars_1_stack_ls = np.concatenate(vars_1_stack_ls)

    return total / n, total_reg / n, total_reg_1 / n, total_cps / n, total_cps_0 / n, total_cps_1 / n, yhat_0, yhat_1, y, means_0_stack_ls, means_1_stack_ls, vars_0_stack_ls, vars_1_stack_ls          

def run_epoch_val(model, dataloader, train, optim, device, save_all=False, block_size=None,  y_mean = 43.3, y_std = 36, samp_fq = 10):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """

    # model.train(False)
    model.train(False)

    total = 0  # total training loss
    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    y = []

    var_hat = []
    var_e = []
    var_a = []

    mean2s_0_stack_ls = []
    var1s_0_stack_ls = []

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:

                # X, outcome = target_val

                y.append(outcome.numpy())
                X = X.to(device)
                outcome = outcome.to(device)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()

                mean1s = []
                mean2s = []
                var1s = []

                for samp_itr in range(samp_fq):
                    all_ouput = model(X)
                    mean1_raw, var1_raw = all_ouput

                    mean1 = mean1_raw.view(-1)
                    var1 = var1_raw.view(-1)

                    mean1s.append(mean1** 2)
                    mean2s.append(mean1)
                    var1s.append(torch.exp(var1))

                # print("mean1s[0].shape", mean1s[0].shape)
                # print(" torch.stack(mean1s, dim=0)[0]", torch.stack(mean1s, dim=0)[:, 0])

                mean2s_0_stack = torch.stack(mean2s, dim=1).to("cpu").detach().numpy()
                mean2s_0_stack_ls.append(mean2s_0_stack)
                var1s_0_stack = torch.stack(var1s, dim=1).to("cpu").detach().numpy()
                var1s_0_stack_ls.append(var1s_0_stack)

                mean1s_ = torch.stack(mean1s, dim=0).mean(dim=0)
                mean2s_ = torch.stack(mean2s, dim=0).mean(dim=0)
                var1s_ = torch.stack(var1s, dim=0).mean(dim=0)

                # print("torch.stack(mean1s, dim=0).shape", torch.stack(mean1s, dim=0).shape)
                # print("mean1s_.shape", mean1s_.shape)
                # print("mean1s_", mean1s_)

                var2 = mean1s_ - mean2s_ ** 2
                var_ = var1s_ + var2
                var_norm = var_ / var_.max()         
                # print("var_", var_)      
                # print("var_.max()", var_.max()) 
                # print("var_norm", var_norm)       
                # exit()

                yhat.append(mean2s_.to("cpu").detach().numpy() * y_std + y_mean)
                var_hat.append(var_norm.to("cpu").detach().numpy())
                var_e.append(var2.to("cpu").detach().numpy())
                var_a.append(var1s_.to("cpu").detach().numpy())

                loss = torch.nn.functional.mse_loss(mean2s_, (outcome - y_mean) / y_std )

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(total / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update()

    yhat = np.concatenate(yhat)
    var_hat = np.concatenate(var_hat)
    var_e = np.concatenate(var_e)
    var_a = np.concatenate(var_a)
    y = np.concatenate(y)

    mean2s_0_stack_ls = np.concatenate(mean2s_0_stack_ls)
    var1s_0_stack_ls = np.concatenate(var1s_0_stack_ls)

    return total / n, yhat, y, var_hat, var_e, var_a, mean2s_0_stack_ls, var1s_0_stack_ls