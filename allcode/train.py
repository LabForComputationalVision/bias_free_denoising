import argparse
import logging
import sys
import torch
import torchvision
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import data, models, utils


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    utils.setup_experiment(args)
    utils.init_logging(args)

    # Build data loaders, a model and an optimizer
    model = models.build_model(args).to(device)
    print(model)
	
    train_loader, valid_loader, test_loader = data.build_dataset(args.dataset, args.data_path, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,60, 70], gamma=0.5)
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_psnr", "valid_ssim"])}
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

    global_step = -1
    for epoch in range(args.num_epochs):
        train_bar = utils.ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()

        for batch_id, inputs in enumerate(train_bar):
            model.train()
            global_step += 1
            inputs = inputs.to(device)
            noise = utils.get_noise(inputs, mode = args.noise_mode, 
                                                min_noise = args.min_noise/255., max_noise = args.max_noise/255.,
                                                noise_std = args.noise_std/255.)

            noisy_inputs = noise + inputs;
            outputs = model(noisy_inputs)
            loss = F.mse_loss(outputs, inputs, reduction="sum") / (inputs.size(0) * 2)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_psnr = utils.psnr(outputs, inputs)
            train_ssim = utils.ssim(outputs, inputs)
            train_meters["train_loss"].update(loss.item())
            train_meters["train_psnr"].update(train_psnr.item())
            train_meters["train_ssim"].update(train_ssim.item())
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("psnr/train", train_psnr.item(), global_step)
                writer.add_scalar("ssim/train", train_ssim.item(), global_step)
                gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
                writer.add_histogram("gradients", gradients, global_step)
                sys.stdout.flush()

        if epoch % args.valid_interval == 0:
            model.eval()
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = utils.ProgressBar(valid_loader)
            for sample_id, sample in enumerate(valid_bar):
                with torch.no_grad():
                    sample = sample.to(device)
                    noise = utils.get_noise(sample, mode = 'S', 
                                                noise_std = (args.min_noise +  args.max_noise)/(2*255.))

                    noisy_inputs = noise + sample;
                    output = model(noisy_inputs)
                    valid_psnr = utils.psnr(output, sample)
                    valid_meters["valid_psnr"].update(valid_psnr.item())
                    valid_ssim = utils.ssim(output, sample)
                    valid_meters["valid_ssim"].update(valid_ssim.item())

                    if writer is not None and sample_id < 10:
                        image = torch.cat([sample, noisy_inputs, output], dim=0)
                        image = torchvision.utils.make_grid(image.clamp(0, 1), nrow=3, normalize=False)
                        writer.add_image(f"valid_samples/{sample_id}", image, global_step)

            if writer is not None:
                writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                writer.add_scalar("ssim/valid", valid_meters['valid_ssim'].avg, global_step)
                sys.stdout.flush()
            if test_loader is not None and writer is not None:
                test_bar = utils.ProgressBar(test_loader)
                for sample_id, sample in enumerate(test_bar):
                    if sample_id >= 10:
                        break
                    with torch.no_grad():
                        sample = sample.to(device)
                        output = model(sample)
                        image = torch.cat([sample, output], dim=0)
                        image = torchvision.utils.make_grid(image.clamp(0, 1), nrow=2, normalize=False)
                        writer.add_image(f"test_samples/{sample_id}", image, global_step)

            logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=optimizer.param_groups[0]["lr"])))
            utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_psnr"].avg, mode="max")
        scheduler.step()

    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="data", help="path to data directory")
    parser.add_argument("--dataset", default="bsd400", help="train dataset name")
    parser.add_argument("--batch-size", default=128, type=int, help="train batch size")

    # Add model arguments
    parser.add_argument("--model", default="dncnn", help="model architecture")
    
    # Add noise arguments
    parser.add_argument("--noise_mode", default="B", help="B - Blind S-one noise level")
    parser.add_argument('--noise_std', default = 25, type = float, 
                help = 'noise level when mode is S')
    parser.add_argument('--min_noise', default = 0, type = float, 
                help = 'minimum noise level when mode is B')
    parser.add_argument('--max_noise', default = 55, type = float, 
                    help = 'maximum noise level when mode is B')

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
#     parser.add_argument("--lr-step-size", default=30, type=int, help="step size for learning rate scheduler")
#     parser.add_argument("--lr-gamma", default=0.1, type=float, help="learning rate multiplier")
    parser.add_argument("--num-epochs", default=70, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[args.model].add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
