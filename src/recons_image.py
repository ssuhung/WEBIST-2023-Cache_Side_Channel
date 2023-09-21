import random
import time

import progressbar
import torch
import torch.nn as nn
from pytorch_msssim import ssim

import models
import utils
from data_loader import *
from loss import SsimLoss
from params import Params
from utils import Printer


class ImageEngine:
    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.ssim = SsimLoss()
        self.real_label = 1
        self.fake_label = 0
        self.init_model_optimizer()
        self.train_losses = []
        self.test_losses = []

    def init_model_optimizer(self):
        # self.enc = models.__dict__['attn_trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
        # self.enc = models.__dict__['trace_encoder_%d' % self.args.trace_w](dim=self.args.nz, nc=self.args.trace_c)
        if self.args.dataset == 'CelebA_jpg':
            self.enc = models.TraceEncoder_1DCNN_encode(input_len=self.args.data_path[args.dataset]['max_trace_len'], dim=self.args.nz)
        elif self.args.dataset == 'CelebA_webp':
            self.enc = models.TraceEncoder_1DCNN_encode_webp(input_len=self.args.data_path[args.dataset]['max_trace_len'], dim=self.args.nz)
        self.enc = self.enc.to(self.args.device)

        self.dec = models.__dict__['ResDecoder%d' % self.args.image_size](dim=self.args.nz, nc=self.args.nc)
        self.dec = self.dec.to(self.args.device)    

        self.optim = torch.optim.Adam(
                        list(self.enc.parameters()) + \
                        list(self.dec.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

        self.E = models.image_output_embed_128(dim=self.args.nz, nc=self.args.nc)
        self.E = self.E.to(self.args.device)

        self.D = models.classifier(dim=self.args.nz, n_class=1, use_bn=False)
        self.D = self.D.to(self.args.device)

        self.C = models.classifier(dim=self.args.nz, n_class=self.args.data_path[args.dataset]['num_class'], use_bn=False)
        self.C = self.C.to(self.args.device)

        self.optim_D = torch.optim.Adam(
                        list(self.E.parameters()) + \
                        list(self.D.parameters()) + \
                        list(self.C.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        Printer.print(f'Save Model to {path}')
        state = {
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'E': self.E.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print(f'Load Model from {path}')
        ckpt = torch.load(path, map_location=self.args.device)
        self.enc.load_state_dict(ckpt['enc'])
        self.dec.load_state_dict(ckpt['dec'])
        self.E.load_state_dict(ckpt['E'])
        self.D.load_state_dict(ckpt['D'])
        self.C.load_state_dict(ckpt['C'])

    def save_state(self, path):
        torch.save({
            'epoch': self.epoch,
            'enc': self.enc.state_dict(),
            'dec': self.dec.state_dict(),
            'E': self.E.state_dict(),
            'D': self.D.state_dict(),
            'C': self.C.state_dict(),
            'optim': self.optim.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'loss': (self.mse, self.l1, self.bce, self.ce, self.ssim),
            'seed': self.args.seed,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
            }, path)

    def load_state(self, path):
        checkpoint = torch.load(path)
        self.enc.load_state_dict(checkpoint['enc'])
        self.dec.load_state_dict(checkpoint['dec'])
        self.E.load_state_dict(checkpoint['E'])
        self.D.load_state_dict(checkpoint['D'])
        self.C.load_state_dict(checkpoint['C'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.optim_D.load_state_dict(checkpoint['optim_D'])
        self.epoch = checkpoint['epoch']
        self.mse, self.l1, self.bce, self.ce, self.ssim = checkpoint['loss']
        self.args.seed = checkpoint['seed']
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        torch.manual_seed(self.args.seed)

    def save_output(self, output, path):
        utils.save_image(output.data, path, normalize=True)

    def zero_grad_G(self):
        self.enc.zero_grad()
        self.dec.zero_grad()
        
    def zero_grad_D(self):
        self.E.zero_grad()
        self.D.zero_grad()
        self.C.zero_grad()

    def set_train(self):
        self.enc.train()
        self.dec.train()
        self.E.train()
        self.D.train()
        self.C.train()

    def set_eval(self):
        self.enc.eval()
        self.dec.eval()
        self.E.eval()
        self.D.eval()
        self.C.eval()

    def _train(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record = utils.Record()
            record_mse = utils.Record()
            record_G = utils.Record()
            record_D = utils.Record()
            record_C_real = utils.Record()
            record_C_fake = utils.Record()
            record_C_real_acc = utils.Record()
            record_C_fake_acc = utils.Record()
            start_time = time.time()
            progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                ID = ID.to(self.args.device)
                bs = image.size(0)

                # train D with real
                self.zero_grad_D()
                real_data = image.to(self.args.device)
                batch_size = real_data.size(0)
                label_real = torch.full((batch_size, 1), self.real_label, dtype=real_data.dtype).to(self.args.device)
                label_fake = torch.full((batch_size, 1), self.fake_label, dtype=real_data.dtype).to(self.args.device)

                embed_real = self.E(real_data)
                output_real = self.D(embed_real)
                errD_real = self.bce(output_real, label_real)
                D_x = output_real.mean().item()

                # train D with fake
                encoded = self.enc(trace)
                noise = torch.randn(bs, self.args.nz).to(self.args.device)
                decoded = self.dec(encoded + 0.05 * noise)
                
                output_fake = self.D(self.E(decoded.detach()))
                errD_fake = self.bce(output_fake, label_fake)
                D_G_z1 = output_fake.mean().item()
                
                errD = errD_real + errD_fake
                
                # train C with real
                pred_real = self.C(embed_real)
                errC_real = self.ce(pred_real, ID)

                (errD_real + errD_fake + errC_real).backward()
                self.optim_D.step()
                record_D.add(errD.item())
                record_C_real.add(errC_real.item())
                record_C_real_acc.add(utils.accuracy(pred_real, ID))

                # train G with D and C
                self.zero_grad_G()

                encoded = self.enc(trace)

                noise = torch.randn(bs, self.args.nz).to(self.args.device)
                decoded = self.dec(encoded + 0.05 * noise)

                embed_fake = self.E(decoded)
                output_fake = self.D(embed_fake)
                pred_fake = self.C(embed_fake)

                errG = self.bce(output_fake, label_real)
                errC_fake = self.ce(pred_fake, ID)
                mse_loss = self.mse(decoded, image)
                ssim_loss = self.ssim(decoded, image)
                # recons_err = 0.84 * mse_loss + ssim_loss
                recons_err = self.args.alpha * ssim_loss + (1 - self.args.alpha) * mse_loss

                (errG + errC_fake + self.args.lambd * recons_err).backward()
                D_G_z2 = output_fake.mean().item()
                self.optim.step()
                record_G.add(errG.item())
                record_mse.add(mse_loss.item())
                record.add(recons_err.item())
                record_C_fake.add(errC_fake.item())
                record_C_fake_acc.add(utils.accuracy(pred_fake, ID))

                if i == 0:
                    self.save_output(decoded, os.path.join(self.args.image_root, (f'train_{self.epoch:03d}.jpg')))
                    self.save_output(image, os.path.join(self.args.image_root, (f'train_{self.epoch:03d}_target.jpg')))

            progress.finish()
            utils.clear_progressbar()
            self.train_losses.append(record.mean())
            Printer.print('----------------------------------------')
            Printer.print(f'Epoch: {self.epoch}')
            Printer.print(f'Costs Time: {(time.time() - start_time):.2f} s')
            Printer.print(f'MSE Loss: {(record_mse.mean()):.6f}')
            Printer.print(f'Recons Loss: {(record.mean()):.6f}')
            Printer.print(f'Loss of G: {(record_G.mean()):.6f}')
            Printer.print(f'Loss of D: {(record_D.mean()):.6f}')
            Printer.print(f'Loss & Acc of C ID real: {(record_C_real.mean()):.6f} & {(record_C_real_acc.mean()):.6f}')
            Printer.print(f'Loss & Acc of C ID fake: {(record_C_fake.mean()):.6f} & {(record_C_fake_acc.mean()):.6f}')
            Printer.print(f'D(x) is: {D_x:.6f}, D(G(z1)) is: {D_G_z1:.6f}, D(G(z2)) is: {D_G_z2:.6f}')

    def train(self, train_loader, test_loader):
        test_freq = self.args.test_freq
        exp_name = self.args.exp_name
        output_root = self.args.output_root
        ckpt_root = self.args.ckpt_root
        
        for i in range(self.epoch, self.args.num_epoch):
            self._train(train_loader)
            if i % test_freq == 0:
                self._test(test_loader)
                self.save_model(os.path.join(ckpt_root, f'{(i+1):03d}.pth'))
            self.save_state(os.path.join(output_root, exp_name, 'temp_state.pth'))
        self.save_model(os.path.join(ckpt_root, 'final.pth'))
        os.remove(os.path.join(output_root, exp_name, 'temp_state.pth'))
        Printer.print('Training finished!')
        self.print_min_loss()

    def _test(self, data_loader):
        self.set_eval()
        record_mse = utils.Record()
        record = utils.Record()
        start_time = time.time()
        progress = progressbar.ProgressBar(maxval=len(data_loader)).start()
        with torch.no_grad():
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)                
                mse_loss = self.mse(decoded, image)
                ssim_loss = self.ssim(decoded, image)
                # recons_err = 0.84 * mse_loss + ssim_loss
                recons_err = self.args.alpha * ssim_loss + (1 - self.args.alpha) * mse_loss
                record_mse.add(mse_loss.item())
                record.add(recons_err.item())

                if i == 0:
                    self.save_output(decoded, os.path.join(self.args.image_root, (f'test_{self.epoch:03d}.jpg')))
                    self.save_output(image, os.path.join(self.args.image_root, (f'test_{self.epoch:03d}_target.jpg')))

            progress.finish()
            utils.clear_progressbar()
            self.test_losses.append(record.mean())
            Printer.print('----------------------------------------')
            Printer.print('Test')
            Printer.print(f'Costs Time: {(time.time() - start_time):.2f} s')
            Printer.print(f'MSE Loss: {(record_mse.mean()):.6f}')
            Printer.print(f'Recons Loss: {(record.mean()):.6f}')

    def print_min_loss(self):
        train_np = np.array(self.train_losses)
        test_np = np.array(self.test_losses)
        Printer.print(f'Minimum training loss: {(train_np.min().item()):.6f} in epoch {train_np.argmin().item() + 1}')
        Printer.print(f'Minimum testing loss: {(test_np.min().item()):.6f} in epoch {test_np.argmin().item() * self.args.test_freq + 1}')

    def save_image(self, output, name_list, path):
        assert len(output) == len(name_list)
        for i in range(len(output)):
            name = name_list[i]
            export_path = os.path.join(path, f'{name}.jpg')
            utils.save_image(output[i].unsqueeze(0).data,
                             export_path,
                             normalize=True, nrow=1, padding=0)
            
    def inference(self, data_loader):
        """
        For the given model,
          1. print the average ssim score
          2. generate reconstruction images
        """
        recons_dir = os.path.join(self.args.output_root, self.args.exp_name, 'recons')
        target_dir = os.path.join(self.args.output_root, self.args.exp_name, 'target')
        utils.make_path(recons_dir)
        utils.make_path(target_dir)

        self.set_eval()
        progress = progressbar.ProgressBar(maxval=len(data_loader), widgets=utils.get_widgets()).start()
        with torch.no_grad():
            ssim_sum = 0
            ssim_cnt = 0
            for i, (trace, image, prefix, ID) in enumerate(data_loader):
                progress.update(i + 1)
                image = image.to(self.args.device)
                trace = trace.to(self.args.device)
                encoded = self.enc(trace)
                decoded = self.dec(encoded)
                decoded = decoded.to('cpu')

                ssim_val = ssim(decoded, image, data_range=1, size_average=True)
                batch_size = image.size(0)
                ssim_sum += ssim_val * batch_size
                ssim_cnt += batch_size

                self.save_image(image, prefix, target_dir)
                self.save_image(decoded, prefix, recons_dir)

        progress.finish()
        utils.clear_progressbar()
        print(f'Avg. SSIM score: {ssim_sum / ssim_cnt}')

if __name__ == '__main__':
    args = Params().parse()

    args.trace_c = 6
    args.trace_w = 256
    args.nz = 128

    args.image_root = os.path.join(args.output_root, args.exp_name, 'image')
    args.ckpt_root = os.path.join(args.output_root, args.exp_name, 'ckpt')
    Printer.output_file = os.path.join(args.output_root, args.exp_name, 'output.out')
    
    engine = ImageEngine(args)

    if os.path.exists(os.path.join(args.output_root, args.exp_name)):
        ans = input(f'Experiment folder "{args.exp_name}" already exist, do you want to continue training or overwrite the result? (continue/overwrite/ctrl+c) ')
        if ans.lower() == 'continue':
            print("Continue training")
            engine.load_state(os.path.join(args.output_root, args.exp_name, 'temp_state.pth'))
        elif ans.lower() == 'overwrite':
            print("Overwrite previous experiment result")
            Printer.print(f'Experiment Name: {args.exp_name}')
            args.seed = random.randint(1, 10000)
            Printer.print('Manual Seed: %d' % args.seed)
            torch.manual_seed(args.seed)
        else:
            print("Unknown input. Program terminate")
            exit(1)
    else:
        os.mkdir(os.path.join(args.output_root, args.exp_name))
        utils.make_path(args.image_root)
        utils.make_path(args.ckpt_root)

        Printer.print(f'Experiment Name: {args.exp_name}')
        args.seed = random.randint(1, 10000)
        Printer.print('Manual Seed: %d' % args.seed)
        torch.manual_seed(args.seed)

    loader = DataLoader(args)
    train_dataset = CelebaDataset(
                    img_dir=args.data_path[args.dataset]['img_dir'], 
                    npz_dir=args.data_path[args.dataset][args.side],
                    ID_path=args.data_path[args.dataset]['ID_path'],
                    split=args.data_path[args.dataset]['split'][0],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    image_size=args.image_size,
                    trace_len=args.data_path[args.dataset]['max_trace_len'],
                    side=args.side,
                    leng=80000,
                    attack=args.attack,
                    img_type=args.data_path[args.dataset]['image_type']
                )
    test_dataset = CelebaDataset(
                    img_dir=args.data_path[args.dataset]['img_dir'], 
                    npz_dir=args.data_path[args.dataset][args.side],
                    ID_path=args.data_path[args.dataset]['ID_path'],
                    split=args.data_path[args.dataset]['split'][1],
                    trace_c=args.trace_c,
                    trace_w=args.trace_w,
                    image_size=args.image_size,
                    trace_len=args.data_path[args.dataset]['max_trace_len'],
                    side=args.side,
                    attack=args.attack,
                    img_type=args.data_path[args.dataset]['image_type']
                )
    train_loader = loader.get_loader(train_dataset)
    test_loader = loader.get_loader(test_dataset, shuffle=False)

    engine.train(train_loader, test_loader)

    # engine.load_model('../output/Pin_cachelineEncode_WebP/ckpt/031.pth')
    # engine.inference(test_loader)
