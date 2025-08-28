import os
import re
import sys
import random
import time
import torch
import pickle
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm
from torchinfo import summary
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from torch.amp import GradScaler, autocast
from segmentation_models_pytorch.metrics import get_stats, iou_score
from stseg.utils import save_losses, create_config


# class CUDAPrefetcher:
#     def __init__(self, loader, device):
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream(device=device)
#         self.device = device
#         self.next = None
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next = next(self.loader)
#         except StopIteration:
#             self.next = None
#             return
#         with torch.cuda.stream(self.stream):
#             self.next['image'] = self.next['image'].to(self.device, non_blocking=True, dtype=torch.float32)
#             self.next['mask'] = self.next['mask'].to(self.device, non_blocking=True, dtype=torch.long)
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         torch.cuda.current_stream(self.device).wait_stream(self.stream)
#         if self.next is None: raise StopIteration
#         item = self.next
#         self.preload()
#         return item


class SegModel:
    def __init__(self, config, print_summary=True):
        self.config = config
        self.print_summary = print_summary

        self.seg_loss = DiceCELoss(softmax=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")

        self.model = smp.create_model(**self.config['model']).to(self.device)

        if self.config['load_model_path'] and 'last' in config['load_model_path'].split('/')[-1]:
            # update loss_dict from previous training, as we are continuing training
            loss_pickle_path = os.path.join(self.config['results_path'], 'loss_dict.pkl')
            if os.path.exists(loss_pickle_path):
                with open(loss_pickle_path, 'rb') as file:
                    self.loss_dict = pickle.load(file)
        else:
            self.loss_dict = {'train_loss': [], 'val_loss': []}

    def train_one_epoch(self, epoch, train_loader, optimizer, scaler):
        self.model.train()
        disable_prog_bar = not self.config['progress_bar']
        epoch_loss = 0

        start = time.time()

        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=150, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            for step, batch in progress_bar:
                images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
                masks = batch["mask"].to(self.device, dtype=torch.long, non_blocking=True)

                with autocast(self.device.type):
                    predictions = self.model(images)

                masks = torch.nn.functional.one_hot(masks, num_classes=self.config['n_classes'] + 1)
                if masks.ndim == 5:
                    masks = masks.permute(0, 4, 1, 2, 3)
                elif masks.ndim == 4:
                    masks = masks.permute(0, 3, 1, 2)

                loss = self.seg_loss(predictions.float(), masks.float())

                scaler.scale(loss).backward()
                if (step + 1) % self.config['grad_accumulate_step'] == 0 or (step + 1) == len(train_loader):
                    # gradient clipping
                    if self.config['grad_clip_max_norm']:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['grad_clip_max_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        epoch_loss /= len(train_loader)

        if disable_prog_bar:
            end = time.time() - start
            print(f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - loss: {epoch_loss:.4f}")

        self.loss_dict["train_loss"].append(epoch_loss)

    def validate_one_epoch(self, epoch, val_loader, return_img_mask_pred=False):
        self.model.eval()
        disable_prog_bar = not self.config['progress_bar']
        epoch_loss = 0

        start = time.time()

        with tqdm(enumerate(val_loader), total=len(val_loader), ncols=150, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
                masks = batch["mask"].to(self.device, dtype=torch.long, non_blocking=True)

                with torch.no_grad():
                    with autocast(self.device.type):
                        predictions = self.model(images)

                masks = torch.nn.functional.one_hot(masks, num_classes=self.config['n_classes'] + 1)
                if masks.ndim == 5:
                    masks = masks.permute(0, 4, 1, 2, 3)
                elif masks.ndim == 4:
                    masks = masks.permute(0, 3, 1, 2)

                loss = self.seg_loss(predictions.float(), masks.float())
                epoch_loss += loss.item()
                progress_bar.set_postfix({"val_loss": epoch_loss / (step + 1)})

        epoch_loss /= len(val_loader)

        if disable_prog_bar:
            end = time.time() - start
            print(f"Inference time: {time.strftime('%H:%M:%S', time.gmtime(end))} - val_loss: {epoch_loss:.4f}")

        self.loss_dict["val_loss"].append(epoch_loss)

        if return_img_mask_pred:
            to_np = lambda x: x.detach().cpu().numpy()
            images, masks, predictions = map(to_np, (images, masks, predictions))
            return images, masks, predictions

    def get_optimizer_and_lr_schedule(self):
        optimizer_class = getattr(torch.optim, self.config['optimizer']['name'])
        optimizer_args = {key: value for key, value in self.config['optimizer'].items() if key != 'name'}
        optimizer = optimizer_class(self.model.parameters(), **optimizer_args)

        if self.config["lr_scheduler"]:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['name'])
            lr_scheduler_args = {key: value for key, value in self.config['lr_scheduler'].items() if key != 'name'}
            lr_scheduler = scheduler_class(optimizer, **lr_scheduler_args)
        else:
            lr_scheduler = None

        return optimizer, lr_scheduler

    def save_model(self, epoch, validation_loss, optimizer, scheduler=None):
        save_path = os.path.join(self.config['results_path'], 'checkpoints')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        last_checkpoint_path = os.path.join(save_path, 'last_model.pth')
        checkpoint = {
            'epoch': epoch,
            'network_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': validation_loss
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, last_checkpoint_path)

        best_checkpoint_path = os.path.join(save_path, 'best_model.pth')
        if os.path.isfile(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
            best_loss = best_checkpoint.get('validation_loss', float('inf'))
            if validation_loss < best_loss:
                torch.save(checkpoint, best_checkpoint_path)
        else:
            torch.save(checkpoint, best_checkpoint_path)

    def load_model(self, load_model_path, optimizer=None, scheduler=None, for_training=False):
        print(f'Loading model from {load_model_path}...')
        checkpoint = torch.load(load_model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['network_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if for_training:
            return checkpoint['epoch'] + 1

    def save_plots(self, images, masks, preds, save_path, one_hot=True):
        if one_hot:
            masks = masks.argmax(1)
            preds = preds.argmax(1)
        rows = min(4, images.shape[0])
        n_classes = self.config['n_classes']

        fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))
        titles = ['image', 'mask', 'prediction']

        for i in range(rows):
            img = images[i].transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            axes[i, 0].imshow(img)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(masks[i], cmap='hot', vmin=0, vmax=n_classes)
            axes[i, 1].axis('off')
            axes[i, 2].imshow(preds[i], cmap='hot', vmin=0, vmax=n_classes)
            axes[i, 2].axis('off')

        for j, t in enumerate(titles):
            axes[0, j].set_title(t)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def train(self, train_loader, val_loader):
        scaler = GradScaler()
        total_start = time.time()
        start_epoch = 1
        plot_save_path = os.path.join(self.config['results_path'], 'plots')

        img_shape = self.config['transformations']['patch_size']
        input_shape = (self.config['batch_size'], 3, *img_shape)
        optimizer, lr_scheduler = self.get_optimizer_and_lr_schedule()

        if self.config['load_model_path']:
            start_epoch = self.load_model(self.config['load_model_path'], optimizer=optimizer, scheduler=lr_scheduler, for_training=True)

        if self.print_summary:
            summary(self.model, input_shape, batch_dim=None, depth=3)

        for epoch in range(start_epoch, self.config['n_epochs'] + 1):
            self.train_one_epoch(epoch, train_loader, optimizer, scaler)
            if epoch % self.config['val_plot_interval'] == 0:
                images, masks, predictions = self.validate_one_epoch(epoch, val_loader, return_img_mask_pred=True)
                self.save_plots(images, masks, predictions, save_path=os.path.join(self.config['results_path'], 'plots', f"epoch_{epoch}.png"))
                del images, masks, predictions
            else:
                self.validate_one_epoch(epoch, val_loader)
            save_losses(self.loss_dict, plot_save_path)
            self.save_model(epoch, self.loss_dict['val_loss'][-1], optimizer, scheduler=lr_scheduler)

            loss_pickle_path = os.path.join(self.config['results_path'], 'loss_dict.pkl')
            with open(loss_pickle_path, 'wb') as file:
                pickle.dump(self.loss_dict, file)

            if lr_scheduler:
                lr_scheduler.step()
                print(f"Adjusting learning rate to {lr_scheduler.get_last_lr()[0]:.4e}.")

        total_time = time.time() - total_start
        print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    def run_inference(self, test_loader):
        # TODO: currently, you might get slightly different results whenever you run this
        torch.backends.cudnn.benchmark = True  # faster convs for fixed sizes
        self.load_model(self.config['load_model_path'])
        self.model.eval()
        # Optional for more speed with 2D convs:
        self.model.to(memory_format=torch.channels_last)
        self.model.half()
        all_iou_list = []

        base_results_path = self.config['results_path']
        prefix = "test_plots_"
        existing = [int(re.search(rf"{prefix}(\d+)", d).group(1))
                    for d in os.listdir(base_results_path) if d.startswith(prefix) and re.search(rf"{prefix}(\d+)", d)]
        plot_folder_path = os.path.join(base_results_path, f"{prefix}{max(existing, default=0) + 1}")
        create_config(self.config, plot_folder_path)
        print('')
        start = time.time()

        seen = set()
        with torch.inference_mode():
            # prefetch_loader = CUDAPrefetcher(test_loader, self.device)
            for data_item in test_loader:
                name = data_item['id']
                frames = data_item['image'].to(self.device, dtype=torch.float16, memory_format=torch.channels_last, non_blocking=True)
                masks = data_item['mask'].to(self.device, dtype=torch.long, non_blocking=True)

                if name not in seen:

                    if len(seen) > 0:
                        video_end = time.time() - video_start
                        print(f"    Inference time: {time.strftime('%H:%M:%S', time.gmtime(video_end))} ({num_frames / video_end:.2f} fps)")
                        video_iou_tensor = torch.cat(current_iou_list, dim=0)
                        mean_per_class = video_iou_tensor.mean(dim=0)
                        all_iou_list.append(video_iou_tensor)
                        parts = []
                        for i, score in enumerate(mean_per_class):
                            parts.append(f"C{i+1}: {score.numpy() * 100:.2f}")
                        parts.append(f"AVG: {mean_per_class.mean().numpy() * 100:.2f}")
                        print("    IOU scores per class: ")
                        print("        " + " - ".join(parts))
                        self.save_plots(*plot_item, one_hot=False)

                    # if len(seen) == 1:
                    #     break

                    current_iou_list = []
                    seen.add(name)
                    vid_idx = test_loader.dataset.ids.index(name)
                    num_frames = test_loader.dataset.video_lengths[vid_idx]
                    print(f"Running inference for: {name}")
                    print(f"    Number of frames: {num_frames}")
                    # randomly select a batch to plot per case
                    plot_batch_t0 = random.choice([t[1] for t in test_loader.dataset._index if t[0] == vid_idx])
                    plot_item = ()
                    video_start = time.time()

                with autocast(self.device.type):
                    preds = sliding_window_inference(frames, roi_size=self.config['transformations']['patch_size'],
                                                     sw_batch_size=self.config['sw_batch_size'], predictor=self.model, overlap=self.config['sw_overlap'],
                                                     sw_device=self.device, device=self.device)
                preds = preds.argmax(1)
                tp, fp, fn, tn = get_stats(preds-1, masks-1, mode='multiclass',
                                           num_classes=self.config['n_classes'], ignore_index=-1)
                score = iou_score(tp, fp, fn, tn)
                current_iou_list.append(score)

                if data_item['t0'] == plot_batch_t0:
                    plot_path = os.path.join(plot_folder_path, f"{name}_{data_item['t0']}-{data_item['t1']}.png")
                    to_np = lambda x: x.detach().cpu().numpy()
                    frames, masks, preds = map(to_np, (frames.float(), masks.long(), preds.long()))
                    plot_item = (frames, masks, preds, plot_path)

            # save results for the last video
            video_end = time.time() - video_start
            print(f"    Inference time: {time.strftime('%H:%M:%S', time.gmtime(video_end))} ({num_frames / video_end:.2f} fps)")
            video_iou_tensor = torch.cat(current_iou_list, dim=0)
            mean_per_class = video_iou_tensor.mean(dim=0)
            all_iou_list.append(video_iou_tensor)
            parts = []
            for i, score in enumerate(mean_per_class):
                parts.append(f"C{i+1}: {score.numpy() * 100:.2f}")
            parts.append(f"AVG: {mean_per_class.mean().numpy() * 100:.2f}")
            print("    IOU scores per class: ")
            print("        " + " - ".join(parts))
            self.save_plots(*plot_item, one_hot=False)

        all_iou_tensors = torch.cat(all_iou_list, dim=0)
        total_mean_per_class = all_iou_tensors.mean(dim=0)

        total = time.time() - start
        print(f"\nTotal inference time: {time.strftime('%H:%M:%S', time.gmtime(total))}")
        parts = []
        for i, score in enumerate(total_mean_per_class):
            parts.append(f"C{i+1}: {score.numpy() * 100:.2f}")
        parts.append(f"AVG: {total_mean_per_class.mean().numpy() * 100:.2f}")
        print("IOU scores per class: ")
        print("    " + " - ".join(parts))

