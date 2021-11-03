import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from lib.net.lookahead import *
from model import *
from dataset import *


matplotlib.use('TkAgg')
#----------------
import torch.cuda.amp as amp
is_amp = True  #True #False

#----------------



def do_valid(net, valid_loader):

    valid_loss = [0,0,0,0]
    valid_num  = 0

    net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        pressure_truth = batch['pressure'].cuda()
        u_out = batch['u_out'].cuda()
        x = batch['feature'].cuda()


        with torch.no_grad():
            with amp.autocast(enabled=is_amp):
                pressure_in, pressure_out  = data_parallel(net, (x))
                pressure = pressure_in*(1-u_out) + pressure_out*u_out
                #loss0 = F.mse_loss(pressure, pressure_truth)
                loss0 = F.l1_loss(pressure, pressure_truth)
                loss1 = mask_l1_loss(pressure_in,  pressure_truth, u_out<0.5)
                loss2 = mask_l1_loss(pressure_out, pressure_truth, u_out>0.5)


        #----------
        valid_num += batch_size
        valid_loss[0] += batch_size*loss0.item()
        valid_loss[1] += batch_size*loss1.item()
        valid_loss[2] += batch_size*loss2.item()

        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)

    assert(valid_num == len(valid_loader.dataset))
    #print('')
    #----------------------
    valid_loss[0] = valid_loss[0]/valid_num
    valid_loss[1] = valid_loss[1]/valid_num
    valid_loss[2] = valid_loss[2]/valid_num
    return valid_loss





# start here ! ###################################################################################


def run_train(

):
    fold = 1
    out_dir = root_dir + '/result/run02/learnable-01/fold%d' % fold
    initial_checkpoint = \
        out_dir + '/checkpoint/00070000.model.pth'  #None #

    #hyper parameters
    start_lr   = 0.0001
    batch_size = 128 #14 #22


    ## setup  ----------------------------------------
    for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(out_dir + '/' + f, exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\t__file__ = %s\n' % __file__)
    log.write('\tout_dir  = %s\n' % out_dir)
    log.write('\n')

    ## dataset ------------------------------------
    df = make_df(mode='train')

    train_idx, valid_idx = make_fold(df, mode='train-1')
    train_dataset = VentilatorDataset(df, train_idx, )#train_augment
    valid_dataset = VentilatorDataset(df, valid_idx, )

    train_loader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = batch_size,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = False,
        worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        #collate_fn  = null_collate,
    )
    valid_loader  = DataLoader(
        valid_dataset,
        sampler = SequentialSampler(valid_dataset),
        batch_size  = 128,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = False,
        #collate_fn  = null_collate,
    )

    log.write('fold  : %d\n'%(fold))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    scaler = amp.GradScaler(enabled = is_amp)
    net = Net(len(x_col)).cuda()


    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict  = f['state_dict']
        net.load_state_dict(state_dict,strict=True)  #True
    else:
        start_iteration = 0
        start_epoch = 0


    log.write('net=%s\n'%(type(net)))
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\n')

    # -----------------------------------------------
    if 0: ##freeze
        for p in net.block0.backbone.parameters(): p.requires_grad = False


    #optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)
    #optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr, momentum=0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr,)

    num_iteration = 500000
    iter_log    = 2500
    iter_valid  = 2500
    iter_save   = list(range(0, num_iteration+1, 2500))#1*1000

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('\n')


    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   fold = %d\n'%(fold))
    log.write('   is_amp = %s \n'%str(is_amp))
    log.write('   batch_size = %d\n'%(batch_size))
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                           |----------- VALID -------------|------------- TRAIN/BATCH -------------\n')
    log.write('rate     iter       epoch  | loss    (in)   (out)    -     |  -      (in)   (out)   | time         \n')
    log.write('---------------------------------------------------------------------------------------------------\n')
              #0.00100  00005000* 10.62   | 0.000   0.602  0.244   0.000  | 0.000   0.267  0.060   |  0 hr 08 min


    def message(mode='print'):
        if mode==('print'):
            asterisk = ' '
            loss = batch_loss
        if mode==('log'):
            asterisk = '*' if iteration in iter_save else ' '
            loss = train_loss

        text = \
            '%0.5f  %08d%s %6.2f  | '%(rate, iteration, asterisk, epoch,) +\
            '%4.3f   %4.3f  %4.3f   %4.3f  | '%(*valid_loss,) +\
            '%4.3f   %4.3f  %4.3f   | '%(*loss,) +\
            '%s' % (time_to_str(timer() - start_timer,'min'))

        return text

    #----
    valid_loss = np.zeros(4,np.float32)
    train_loss = np.zeros(3,np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0

    loss0 = torch.FloatTensor([0]).cuda().sum()
    loss1 = torch.FloatTensor([0]).cuda().sum()
    loss2 = torch.FloatTensor([0]).cuda().sum()


    start_timer = timer()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0
    while  iteration < num_iteration:

        for t, batch in enumerate(train_loader):
            if iteration in iter_save:
                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d.model.pth' % (iteration))
                    pass

            if (iteration % iter_valid == 0):
                #if iteration!=start_iteration:
                    valid_loss = do_valid(net, valid_loader)  #
                    pass

            if (iteration % iter_log == 0):
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')


            # learning rate schduler ------------
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            batch_size = len(batch['index'])
            pressure_truth = batch['pressure'].cuda()
            u_out = batch['u_out'].cuda()
            x = batch['feature'].cuda()

            #----
            net.train()
            optimizer.zero_grad()

            with amp.autocast(enabled = is_amp):
                pressure_in, pressure_out  = data_parallel(net, (x))
                # loss0 = F.mse_loss(pressure, pressure_truth)
                #loss0 = F.huber_loss(pressure, pressure_truth)

                loss1 = mask_smooth_l1_loss(pressure_in, pressure_truth, u_out<0.5)
                loss2 = mask_smooth_l1_loss(pressure_out, pressure_truth, u_out>0.5)
                #loss1 = mask_l1_loss(pressure, pressure_truth, u_out)

                #F.l1_loss
                #F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)




            scaler.scale(loss0+loss1+loss2).backward()
            scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()


            # print statistics  --------
            epoch += 1 / len(train_loader)
            iteration += 1

            batch_loss = np.array([loss0.item(), loss1.item(), loss2.item()])
            sum_train_loss += batch_loss
            sum_train += 1
            if iteration % min(iter_log,100) == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)

    log.write('\n')





# main #################################################################
if __name__ == '__main__':
    run_train()