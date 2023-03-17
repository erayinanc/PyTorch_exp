#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to train Convolutional Defiltering Model (CDM)
based on Denoising Diffusion Probabilistic Model (DDPM)
with MNIST dataset

authors: EI
version: 230315a
prereq: python3.x w/ torch, torchvision, matplotlib, numpy, scipy, perlin_noise

notes: bases on the CNN of MNIST example: https://github.com/pytorch/examples/blob/main/mnist/main.py
CDM defilters heavily filtered data to generate samples (initial thoughts for advanced applications in CFD)
training is done on a system with m1 chip from Apple

help: ./MNIST_CDM.py --help
"""

# remove torchvision warnings on macos
import warnings
warnings.filterwarnings("ignore")

# std libs
import argparse, sys, os, time, numpy as np, logging, random, matplotlib.pyplot as plt

# ml libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from networks import mini_U_Net, diff_model_2d, noise_gen_2d

# parsed settings
def pars_ini():
    global args
    parser = argparse.ArgumentParser(description='Train MNIST with CDM model to test possible image generation -- experimental')

    # I/O
    parser.add_argument('--data-dir', default='./',
                        help='location of the training dataset in the'
                        ' local filesystem (default: ./)')
    parser.add_argument('--restart-int', type=int, default=10,
                        help='restart interval per epoch (default: 10)')
    parser.add_argument('--concM', type=int, default=1,
                        help='increase dataset size with this factor (default: 1)')

    # model
    parser.add_argument('--batch-size', type=int, default=96, choices=range(1,int(1e7)), metavar="[1-1e9]",
                        help='input batch size for training (default: 96, min: 1, max: 1e9)')
    parser.add_argument('--epochs', type=int, default=10, choices=range(1,int(1e7)), metavar="[1-1e9]",
                        help='number of epochs to train (default: 10, min: 1, max: 1e9)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--wdecay', type=float, default=0.003,
                        help='weight decay in Adam optimizer (default: 0.003)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='gamma in schedular (default: 0.95)')
    parser.add_argument('--shuff', action='store_true', default=True,
                        help='shuffle dataset p/ epoch (default: True)')
    parser.add_argument('--schedule', action='store_true', default=True,
                        help='enable scheduler in the training (default: True)')

    # debug parsers
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run with seed (default: False)')
    parser.add_argument('--skipplot', action='store_true', default=False,
                        help='skips test postprocessing (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training (default: 10)')

    # optimization
    parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
    parser.add_argument('--nworker', type=int, default=0,
                        help='number of workers in DataLoader (default: 0 - only main)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--accum-iter', type=int, default=1,
                        help='accumulate gradient update (default: 1 - turns off)')

    # benchmarking
    parser.add_argument('--synt', action='store_true', default=False,
                        help='use a synthetic dataset instead (default: False)')
    parser.add_argument('--synt-dpw', type=int, default=3, choices=range(1,int(1e7)), metavar="[1-1e9]",
                        help='dataset size per worker if synt is true (default: 3, min: 1, max: 1e9)')
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')

    args = parser.parse_args()

# debug of the run
def debug_ini(timer):
    logging.basicConfig(format='%(levelname)s: %(message)s', stream=sys.stdout, level=logging.INFO)
    logging.info('configuration:')
    logging.info('sys.version: '+str(sys.version))
    logging.info('parsers list:')
    list_args = [x for x in vars(args)]
    for count,name_args in enumerate(list_args):
        logging.info('args.'+name_args+': '+str(vars(args)[list_args[count]]))

    # add warning here!
    warning1=False
    print(f'\n--------------------------------------------------------')
    if args.benchrun and args.epochs<3:
        logging.warning('benchrun requires atleast 3 epochs - setting epochs to 3!')

        # set minimum of 3 epochs when benchmarking (last epoch produces logs)
        args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs
        warning1=True
    if not args.mps and torch.backends.mps.is_available():
        logging.warning('Found mps device, please run with --mps to enable mac GPU, using CPUs for now!')
        warning1=True
    if not warning1:
        logging.warning('all OK!')
    print(f'--------------------------------------------------------\n')

    return logging

# debug of the training
def debug_final(logging,start_epoch,last_epoch,first_ep_t,last_ep_t,tot_ep_t):
    done_epochs = last_epoch - start_epoch + 1
    print(f'\n--------------------------------------------------------')
    logging.info('training results:')
    logging.info('first epoch time: {:.2f}'.format(first_ep_t)+' s')
    logging.info('last epoch time: {:.2f}'.format(last_ep_t)+' s')
    logging.info('total epoch time: {:.2f}'.format(tot_ep_t)+' s')
    logging.info('average epoch time: {:.2f}'.format(tot_ep_t/done_epochs)+' s')
    if done_epochs>1:
        tot_ep_tm1 = tot_ep_t - first_ep_t
        logging.info('total epoch-1 time: {:.2f}'.format(tot_ep_tm1)+' s')
        logging.info('average epoch-1 time: {:.2f}'.format(tot_ep_tm1/(done_epochs-1))+' s')
    if args.benchrun and done_epochs>2:
        tot_ep_tm2 = tot_ep_t - first_ep_t - last_ep_t
        logging.info('total epoch-2 time: {:.2f}'.format(tot_ep_tm2)+' s')
        logging.info('average epoch-2 time: {:.2f}'.format(tot_ep_tm2/(done_epochs-2))+' s')

# save state of the training
def save_state(epoch,model,loss_acc,optimizer,res_name,is_best):
    rt = time.perf_counter()

    # collect state
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'best_acc': loss_acc,
             'optimizer' : optimizer.state_dict()}

    # write on worker with is_best
    torch.save(state,'./'+res_name)
    logging.info('state is saved on epoch:'+str(epoch)+\
            ' in {:.2f}'.format(time.perf_counter()-rt)+' s')

# custom loss function
class custom_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # constants
        self.b1 = 0.5
        self.b2 = 0.01

    def forward(self,inputs,targets):
        # loss pixel with MSE
        L_1 = self.b1*torch.mean((inputs-targets)**2.0)

        # loss gradient with MSE
        ix = torch.gradient(inputs,dim=[2])[0]
        iy = torch.gradient(inputs,dim=[3])[0]
        tx = torch.gradient(targets,dim=[2])[0]
        ty = torch.gradient(targets,dim=[3])[0]
        L_2  = self.b2*torch.mean((ix-tx)**2.0)
        L_2 += self.b2*torch.mean((iy-ty)**2.0)

        # normalise L2 wrt L1 so each L are in the same magnitude
        return L_1 + L_2 / 10.0**torch.ceil(torch.log10(L_2/L_1))

# synthetic data for benchmarking
class SyntheticDataset_train(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        data = torch.randn(1, 28, 28)
        target = random.randint(0, 999)
        return (data, target)

    def __len__(self):
        return args.batch_size * args.synt_dpw

class SyntheticDataset_test(SyntheticDataset_train):
    def __len__(self):
        return args.batch_size

# deterministic dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def trace_handler(prof):
    # do operations when a profiler calles a trace
    #prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")
    logging.info('profiler called a trace')

# train loop
def train(model, device, train_loader, optimizer, epoch, loss_function, scheduler, noise_map):
    # start a timer
    lt_1 = time.perf_counter()

    # profiler
    """
    - activities (iterable): list of activity groups (CPU, CUDA) to use in profiling,
    supported values: torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA.
    Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA.
    - schedule (callable): callable that takes step (int) as a single parameter and returns
    ProfilerAction value that specifies the profiler action to perform at each step.
        the profiler will skip the first ``skip_first`` steps,
        then wait for ``wait`` steps,
        then do the warmup for the next ``warmup`` steps,
        then do the active recording for the next ``active`` steps and
        then repeat the cycle starting with ``wait`` steps.
        The optional number of cycles is specified with the ``repeat`` parameter,
           0 means that the cycles will continue until the profiling is finished.
    - on_trace_ready (callable): callable that is called at each step
    when schedule returns ProfilerAction.RECORD_AND_SAVE during the profiling.
    - record_shapes (bool): save information about operator's input shapes.
    - profile_memory (bool): track tensor memory allocation/deallocation.
    - with_stack (bool): record source information (file and line number) for the ops.
    - with_flops (bool): use formula to estimate the FLOPs (floating point operations)
    of specific operators (matrix multiplication and 2D convolution).
    - with_modules (bool): record module hierarchy (including function names) corresponding
    to the callstack of the op. e.g. If module A's forward call's module B's forward
    which contains an aten::add op, then aten::add's module hierarchy is A.B
    Note that this support exist, at the moment, only for TorchScript models and not eager mode models.
    """
    if args.benchrun:
        # profiler options
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ],
            # at least 3 epochs required with
            # default wait=1, warmup=1, active=args.epochs, repeat=1, skip_first=0
            schedule=torch.profiler.schedule(wait=1,warmup=1,active=args.epochs,repeat=1,skip_first=0),
            #on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            on_trace_ready=trace_handler,
            record_shapes=False,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=False
        )
        # profiler start
        prof.start()

    loss_acc=0.0
    lt_2 = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        do_backprop = ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(train_loader))

        # diffusion model
        lt_3 = time.perf_counter()
        res = diff_model_2d(data, noise_map, epoch)
        lt_2 += time.perf_counter() - lt_3

        with torch.set_grad_enabled(True):
            # forward pass
            output = model(res.datas.to(device)).float()
            loss = loss_function(output, data.to(device)) / args.accum_iter

            # backward pass
            loss.backward()
            if do_backprop:
                optimizer.step()
                optimizer.zero_grad()

        loss_acc += loss.item()

        if batch_idx % args.log_int == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(data):6d}/{len(train_loader.dataset)} '
                  f'({100.0 * batch_idx / len(train_loader):2.0f}%)]\tloss: {loss_acc:.6f}'
                  f' / sigma={res.sigma}', end='')
            print(f' / bp: {do_backprop}') if not do_backprop else print(f'')

        # profiler step per batch
        if args.benchrun:
            prof.step()

    # TEST w/ plots
    if epoch%1==0:
        plot_scatter_test(data[0,0,:,:].detach().cpu().numpy(), \
                res.datas[0,0,:,:].detach().cpu().numpy(), \
                output[0,0,:,:].detach().cpu().numpy(), epoch, res.sigma)

    # lr scheduler
    if args.schedule:
        scheduler.step()

    # profiler end
    if args.benchrun:
        prof.stop()

    # timer for current epoch
    logging.info('accumulated lost: {:19.16f}'.format(loss_acc))
    logging.info('epoch time: {:.2f}'.format(time.perf_counter()-lt_1)+' s')
    logging.info('filter time: {:.2f}'.format(lt_2)+' s ({:3.2f}'.\
            format(100*lt_2/(time.perf_counter()-lt_1))+'% of epoch time)\n')

    # printout profiler
    if args.benchrun and epoch==args.epochs-1:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: benchmark of last epoch:\n')
        print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=-1))

    return loss_acc, time.perf_counter()-lt_1

# test loop
def test(model, device, test_loader, loss_function, noise_map):
    # start a timer
    lt = time.perf_counter()

    for sigma_test in [1,2,3,4,5,10]:
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # test highly diffusive data
                res = diff_model_2d(data, noise_map, sigma=sigma_test)
                output = model(res.datas.to(device)).float()
                test_loss += loss_function(output, data.to(device)) / args.accum_iter

        # test results
        logging.info('testing results:')
        logging.info('test loss: '+str(test_loss.cpu().numpy()))

        # plot comparison if needed
        if not args.skipplot and not args.testrun and not args.benchrun:
            plot_scatter_test(data[0,0,:,:].detach().cpu().numpy(), \
                    res.datas[0,0,:,:].detach().cpu().numpy(), \
                    output[0,0,:,:].detach().cpu().numpy(), \
                    epoch=args.epochs, sigma=res.sigma, final=True)

    logging.info('total testing time: {:.2f}'.format(time.perf_counter()-lt)+' s')

# post-process
def plot_scatter_test(inp_img, org_img, out_img, epoch, sigma, final=False):
    fig = plt.figure(figsize = (4,12))
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(inp_img, vmin = np.min(inp_img), vmax = np.max(inp_img), interpolation='None')
    ax1.set_title('Original')
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(org_img, vmin = np.min(inp_img), vmax = np.max(inp_img), interpolation='None')
    ax2.set_title('Input')
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(out_img, vmin = np.min(inp_img), vmax = np.max(inp_img), interpolation='None')
    ax3.set_title('Output')
    fig.subplots_adjust(right=0.85)
    fig.tight_layout(pad=1.0)
    if final:
        resName='recon_test_'+str(int(sigma))+'.png'
    else:
        resName='recon_train_'+str(epoch)+'_'+str(int(sigma))+'.png'
    plt.savefig(resName,bbox_inches='tight',pad_inches=0)

def main():
    # get parse args
    pars_ini()

    # start the time.time for profiling
    st = time.perf_counter()

    # deterministic testrun
    if args.testrun:
        torch.manual_seed(args.nseed)
        g = torch.Generator()
        g.manual_seed(args.nseed)

    # debug of the run
    logging = debug_ini(time.perf_counter()-st)

    # set device to CPU
    device = torch.device('mps' if args.mps and torch.backends.mps.is_available() else 'cpu')

    # define train/test data
    data_dir = args.data_dir
    mnist_scale = args.concM
    largeData = []
    for i in range(mnist_scale):
        largeData.append(
            datasets.MNIST(data_dir, train=True, download=True,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
            )
    # concat data (to increase dataset size for testing purposes)
    train_dataset = torch.utils.data.ConcatDataset(largeData)

    mnist_scale = args.concM
    largeData = []
    for i in range(mnist_scale):
        largeData.append(
            datasets.MNIST(data_dir, train=False, download=False,
            transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
            )
    # concat data (to increase dataset size for testing purposes)
    test_dataset = torch.utils.data.ConcatDataset(largeData)

    # instead use a syntheticly generated dataset (good for benchmarks)
    if args.synt:
        # synthetic dataset if selected
        train_dataset = SyntheticDataset_train()
        test_dataset = SyntheticDataset_test()

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

    # load data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        num_workers=args.nworker, pin_memory=True, shuffle=args.shuff, prefetch_factor=args.prefetch, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
        num_workers=args.nworker, pin_memory=True, shuffle=args.shuff, prefetch_factor=args.prefetch, **kwargs)

    # create CNN model
    model = mini_U_Net().to(device)

    # optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # loss function
    loss_function = custom_loss()

    # scheduler
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    # used lr and info on num. of parameters
    tp_d = sum(p.numel() for p in model.parameters())
    logging.info('total distributed parameters: '+str(tp_d))
    tpt_d = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('total distributed trainable parameters: '+str(tpt_d)+'\n')

    # resume state
    start_epoch = 1
    best_acc = np.Inf
    res_name='checkpoint.pth.tar'
    if os.path.isfile(res_name) and not args.benchrun:
        try:
            # Map model to be loaded to specified single gpu.
            loc = {'cpu:%d' % 0: 'cpu:%d' % 0}
            checkpoint = torch.load('./'+res_name, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.warning('restarting from #'+str(start_epoch)+' epoch!')
        except ValueError:
            logging.warning('restart file cannot be loaded, starting from 1st epoch!')

    only_test = start_epoch>args.epochs
    if only_test:
        logging.warning('given epochs are less than the one in the restart file!')
        logging.warning('only testing will be performed -- skipping training!\n')
    else:
        logging.info('starting the training!')
        print(f'--------------------------------------------------------')

    # preprocess noise map
    et = time.perf_counter()
    a,b = next(iter(train_loader))[0].size()[2::]
    noise_map = noise_gen_2d(a,b)
    logging.info('noise generated in: {:.2f}'.format(time.perf_counter()-et)+' s\n')

    # start trainin loop
    et = time.perf_counter()
    first_ep_t = last_ep_t = tot_ep_t = 0.0
    with open('out_loss.dat','w',encoding="utf-8") as outT:
        for epoch in range(start_epoch, args.epochs+1):
            # training
            loss_acc, train_t = train(model, device, train_loader, optimizer, epoch, \
                    loss_function, scheduler, noise_map)

            # save total/first/last epoch timer
            tot_ep_t += train_t
            if epoch == 1:
                first_ep_t = train_t
            if epoch == args.epochs:
                last_ep_t = train_t

           # save state if found a better state
            is_best = loss_acc < best_acc
            if epoch % args.restart_int == 0 and not args.benchrun:
                save_state(epoch, model, loss_acc, optimizer, res_name, is_best)
                # reset best_acc
                best_acc = min(loss_acc, best_acc)

            # write out loss and epoch
            outT.write("%4d   %5.15E\n" %(epoch, loss_acc))
            outT.flush()

    # finalise training
    # save final state
    if not args.benchrun and not only_test:
        print('\nsaving final model!')
        save_state(epoch, model, loss_acc, optimizer, res_name, True)

    # debug final results
    if not only_test:
        debug_final(logging, start_epoch, epoch, first_ep_t, last_ep_t, tot_ep_t)

    # start testing loop
    test(model, device, test_loader, loss_function, noise_map)

    # print duration
    logging.info('final time: {:.2f}'.format(time.perf_counter()-st)+' s')

if __name__ == "__main__":
    main()
    sys.exit()

#eof