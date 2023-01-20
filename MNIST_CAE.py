# -*- coding: utf-8 -*-
"""
script to train a CAE model with MNIST dataset
authors: EI
version: 230120a
notes: bases on the CNN of MNIST example: https://github.com/pytorch/examples/blob/main/mnist/main.py
training is done on a system with m1 chip from Apple
"""

# remove torchvision warnings on macos
import warnings
warnings.filterwarnings("ignore")

# std libs
import argparse, sys, os, time, numpy as np, logging, matplotlib.pyplot as plt, h5py

# ml libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# parsed settings
def pars_ini():
    global args
    parser = argparse.ArgumentParser(description='Train MNIST with CAE model to compress the dataset -- experimental')

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
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wdecay', type=float, default=0.003,
                        help='weight decay in Adam optimizer (default: 0.003)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='gamma in schedular (default: 0.95)')
    parser.add_argument('--shuff', action='store_true', default=False,
                        help='shuffle dataset p/ epoch (default: True)')
    parser.add_argument('--schedule', action='store_true', default=True,
                        help='enable scheduler in the training (default: False)')

    # debug parsers
    parser.add_argument('--testrun', action='store_true', default=False,
                        help='do a test run with seed (default: False)')
    parser.add_argument('--skipplot', action='store_true', default=False,
                        help='skips test postprocessing (default: False)')
    parser.add_argument('--export-latent', action='store_true', default=False,
                        help='export the latent space on testing for debug (default: False)')
    parser.add_argument('--nseed', type=int, default=0,
                        help='seed integer for reproducibility (default: 0)')
    parser.add_argument('--log-int', type=int, default=10,
                        help='log interval per training (default: 10)')

    # optimization
    parser.add_argument('--nworker', type=int, default=0,
                        help='number of workers in DataLoader (default: 0 - only main)')
    parser.add_argument('--prefetch', type=int, default=2,
                        help='prefetch data in DataLoader (default: 2)')
    parser.add_argument('--accum-iter', type=int, default=1,
                        help='accumulate gradient update (default: 1 - turns off)')

    # benchmarking
    parser.add_argument('--benchrun', action='store_true', default=False,
                        help='do a bench run w/o IO (default: False)')

    args = parser.parse_args()

    # set minimum of 3 epochs when benchmarking (last epoch produces logs)
    args.epochs = 3 if args.epochs < 3 and args.benchrun else args.epochs

# debug of the run
def debug_ini(timer):
    logging.basicConfig(format='%(levelname)s: %(message)s', stream=sys.stdout, level=logging.INFO)
    logging.info('configuration:')
    logging.info('--------------------------------------------------------')
    logging.info('initialise in {:.2f}'.format(timer)+' s')
    logging.info('local workers: '+str(args.nworker))
    logging.info('sys.version: '+str(sys.version))
    logging.info('parsers list:')
    list_args = [x for x in vars(args)]
    for count,name_args in enumerate(list_args):
        logging.info('args.'+name_args+': '+str(vars(args)[list_args[count]]))

    # add warning here!
    warning1=False
    if args.benchrun and args.epochs<3:
        logging.warning('benchrun requires atleast 3 epochs - setting epochs to 3\n')
        warning1=True
    if not warning1:
        logging.warning('all OK!\n')

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

# network vae
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(20)

        self.conv3 = nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(10)
        self.conv4 = nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # encoder
        conv1 = self.leaky_reLU(self.bn1(self.conv1(x)))
        conv2 = self.leaky_reLU(self.bn2(self.conv2(conv1)))

        # decoder
        conv3 = self.leaky_reLU(self.bn3(self.conv3(conv2)))
        return self.conv4(conv3)

# compression part - export latent space
class encoder(VAE):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # only encoder part
        conv1 = self.leaky_reLU(self.bn1(self.conv1(x)))
        return self.leaky_reLU(self.bn2(self.conv2(conv1)))

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
def train(model, device, train_loader, optimizer, epoch, loss_function, scheduler):
    # start a timer
    lt = time.perf_counter()

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

    loss_acc=0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        do_backprop = ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(train_loader))
        data, target = data.to(device), target.to(device)
        with torch.set_grad_enabled(True):
            # forward pass
            output = model(data)
            loss = loss_function(output, data)

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % args.log_int == 0:
            print(f'Train epoch: {epoch} [{batch_idx * len(data):6d}/{len(train_loader.dataset)} '
                f'({100.0 * batch_idx / len(train_loader):2.0f}%)]\t\tLoss: {loss.item():.6f}')
        loss_acc+= loss.item()

        # profiler step per batch
        if args.benchrun:
            prof.step()

    # lr scheduler
    if args.schedule:
        scheduler.step()

    # profiler end
    if args.benchrun:
        prof.stop()

    # timer for current epoch
    logging.info('epoch time: {:.2f}'.format(time.perf_counter()-lt)+' s\n')

    # printout profiler
    if args.benchrun and epoch==args.epochs-1:
        print(f'\n--------------------------------------------------------')
        print(f'DEBUG: benchmark of last epoch:\n')
        print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=-1))

    return loss_acc, time.perf_counter()-lt

# test loop
def test(model, device, test_loader, loss_function):
    # start a timer
    lt = time.perf_counter()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, data)

    # test results
    logging.info('testing results:')
    logging.info('total testing time: {:.2f}'.format(time.perf_counter()-lt)+' s')
    logging.info('test_loss: '+str(test_loss.numpy()))

    # plot comparison if needed
    if not args.skipplot and not args.testrun and not args.benchrun:
        plot_scatter(data[0,0,:,:].numpy(), output[0,0,:,:].numpy(), 'test')

# encode export
def encode_exp(encode, device, train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = encode(data).float()

        # export the data
        ini = data.to(device).numpy()
        res = output.to(device).detach().numpy()
        h5f = h5py.File('./latent.h5', 'w')
        h5f.create_dataset('ini', data=ini)
        h5f.create_dataset('res', data=res)
        h5f.close()
        logging.info('latent space is exported to latent.h5')
        break

# plot reconstruction
def plot_scatter(inp_img, out_img, data_org):
    fig = plt.figure(figsize = (4,8))
    ax1 = fig.add_subplot(121)
    im1 = ax1.imshow(inp_img, vmin=np.min(inp_img), vmax=np.max(inp_img), interpolation='None')
    ax1.set_title('Input')
    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(out_img, vmin=np.min(inp_img), vmax=np.max(inp_img), interpolation='None')
    ax2.set_title('Output')
    fig.subplots_adjust(right=0.85)
    fig.tight_layout(pad=1.0)
    plt.savefig('recon_CAE.png', bbox_inches = 'tight', pad_inches = 0)

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
    device = torch.device('cpu')

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

    # deterministic testrun - the same dataset each run
    kwargs = {'worker_init_fn': seed_worker, 'generator': g} if args.testrun else {}

    # load data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        num_workers=args.nworker, pin_memory=True, shuffle=args.shuff, prefetch_factor=args.prefetch, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
        num_workers=args.nworker, pin_memory=True, shuffle=args.shuff, prefetch_factor=args.prefetch, **kwargs)

    # create CNN model
    model = VAE().to(device)

    # optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    # loss function
    loss_function = nn.MSELoss()

    # scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

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
        logging.info('--------------------------------------------------------')

    # printout loss and epoch
    if not only_test:
        outT = open('out_loss.dat','w')

    # start trainin loop
    et = time.perf_counter()
    tot_ep_t = 0.0
    for epoch in range(start_epoch, args.epochs+1):
        # training
        loss_acc, train_t = train(model, device, train_loader, optimizer, epoch, loss_function, scheduler)

        # save total/first/last epoch timer
        tot_ep_t += train_t
        if epoch == start_epoch:
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

    # close file
    if not only_test:
        outT.close()

    # finalise training
    # save final state
    if not args.benchrun and not only_test:
        save_state(epoch, model, loss_acc, optimizer, res_name, True)

    # debug final results
    if not only_test:
        debug_final(logging, start_epoch, epoch, first_ep_t, last_ep_t, tot_ep_t)

    # start testing loop
    test(model, device, test_loader, loss_function)

    # export first batch's latent space if selected
    if args.export_latent:
        encode = encoder().to(device)
        encode_exp(encode, device, train_loader)

    # print duration
    logging.info('final time: {:.2f}'.format(time.perf_counter()-st)+' s')

if __name__ == "__main__":
    main()
    sys.exit()

#eof
