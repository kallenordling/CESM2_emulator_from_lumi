import os
import random
from typing import Any, Callable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from accelerate import Accelerator
from diffusers import SchedulerMixin
from omegaconf.dictconfig import DictConfig
from torch.optim import Optimizer
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import reduce
# import wandb
from ema_pytorch import EMA
# from utils.viz_utils import create_gif
from data.climate_dataset import ClimateDataset, ClimateDataLoader
from models.video_net import UNetModel3D
# from utils.gen_utils import generate_samples
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from torch.serialization import add_safe_globals
import ema_pytorch


def _get_ema_state_dict(ema_obj):
    """
    Works for both ema-pytorch.EMA and custom EMA wrappers.
    Tries common attributes in this order.
    """
    if ema_obj is None:
        return None
    # ema-pytorch exposes .state_dict()
    if hasattr(ema_obj, "state_dict"):
        return ema_obj.state_dict()
    # some wrappers store the averaged model as .ema_model
    if hasattr(ema_obj, "ema_model") and hasattr(ema_obj.ema_model, "state_dict"):
        return ema_obj.ema_model.state_dict()
    raise AttributeError("EMA object does not expose a state_dict().")


def _load_ema_state_dict(ema_obj, state):
    """
    Load EMA state into the existing EMA object.
    Supports ema-pytorch.EMA (has .load_state_dict) and
    fallback to .ema_model.load_state_dict.
    """
    if ema_obj is None or state is None:
        return
    if hasattr(ema_obj, "load_state_dict"):
        ema_obj.load_state_dict(state)
        return
    if hasattr(ema_obj, "ema_model") and hasattr(ema_obj.ema_model, "load_state_dict"):
        ema_obj.ema_model.load_state_dict(state)
        return
    raise AttributeError("EMA object does not support load_state_dict().")


def _list_ckpts_sorted(ckpt_dir, pattern="ckpt_epoch_*.pt"):
    import os, re, glob
    paths = glob.glob(os.path.join(ckpt_dir, pattern))  # <-- module.function

    def _key(p):
        m = re.search(r"epoch[_-](\d+)", os.path.basename(p))
        return (int(m.group(1)) if m else -1, os.path.getmtime(p))

    return sorted(paths, key=_key, reverse=True)


def calc_mse_loss(model_output, target, lats):
    """Manually calculate mse loss"""
    spatial_loss = (model_output - target) ** 2

    # Weight the equator more heavily than the poles
    latitude = torch.as_tensor(lats.values, dtype=spatial_loss.dtype, device=spatial_loss.device)

    latitude_rad = torch.deg2rad(latitude)
    latitude_weight = torch.cos(latitude_rad)

    # Weight the loss
    # print(spatial_loss.shape,latitude_weight.shape)
    lat_weighted_loss = torch.einsum('...yx,y->...yx', spatial_loss,
                                     latitude_weight).mean()  # (spatial_loss * latitude_weight).mean()

    return lat_weighted_loss


class UNetTrainer:
    """Trainer class for 2D diffusion models."""

    def __init__(
            self,
            train_set: ClimateDataset,
            model: UNetModel3D,
            scheduler: SchedulerMixin,
            accelerator: Accelerator,
            hyperparameters: DictConfig,
            dataloader: Callable[[Any], DataLoader],
            optimizer: Callable[[Any], Optimizer],
    ) -> None:
        # Assign the hyperparameters to class attributes
        self.save_hyperparameters(hyperparameters)

        # Assign more class attributes
        self.accelerator = accelerator
        self.train_set, self.val_set = train_set, 0
        self.model = model
        self.scheduler: SchedulerMixin = scheduler
        self.cond_loss_scaling = 0.2
        self.scheduler.set_timesteps(self.sample_steps)

        # Keep track of our exponential moving average weights
        self.ema_model = EMA(
            self.model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10,
        ).to(self.accelerator.device)

        # Assign the device and weight dtype (32 bit for training)
        self.device = self.accelerator.device
        self.weight_dtype = torch.float32

        self.optimizer = optimizer(
            self.model.parameters(), lr=self.lr
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=50,  # Was 20 → Wait longer
            min_lr=1e-6,  # Was 1e-7 → Don't go too small
            cooldown=20,  # Add this
            threshold=0.0001  # Add this
        )
        self.train_loader: ClimateDataLoader = dataloader(
            self.train_set,
            self.accelerator,
            self.batch_size,
            # shuffle=True,
            # drop_last=True,              # avoid short last batch on any rank
            # pin_memory=True,
            # persistent_workers=True,num_workers=4
        )
        # self.val_loader: ClimateDataLoader = dataloader(
        #    self.val_set,
        #    self.accelerator,
        #    self.batch_size,
        # )

        # Initialize counters
        self.global_step = 0
        self.first_epoch = 0

        # Best model tracking
        self.best_loss = float('inf')
        self.best_epoch = -1

        # Keep track of important variables for logging
        self.total_batch_size = (
                self.batch_size
                * self.accelerator.num_processes
                * self.accelerator.gradient_accumulation_steps
        )
        self.num_steps_per_epoch = (
                len(self.train_loader)
                // self.accelerator.gradient_accumulation_steps
                // self.accelerator.num_processes
        )
        self.max_train_steps = self.max_epochs * self.num_steps_per_epoch

        # Log to WANDB (on main process only)
        if self.accelerator.is_main_process:
            self.log_hparams()

        # Load model states from checkpoints if they exist
        if self.load_path:
            self.load(self.load_path)

        # Prepare everything for GPU training
        self.prepare()

    def save_hyperparameters(self, cfg: DictConfig) -> None:
        """Saves the hyperparameters as class attributes."""
        for key, value in cfg.items():
            setattr(self, key, value)

    def log_hparams(self):
        """Logs the hyperparameters to WANDB."""
        # run = self.accelerator.get_tracker("wandb").tracker

        hparam_dict = {
            "Number Training Examples": len(self.train_set)
                                        * len(self.train_set.realizations),
            "Number Epochs": self.max_epochs,
            "Batch Size per Device": self.batch_size,
            "Total Train Batch Size (w. distributed & accumulation)": self.total_batch_size,
            "Gradient Accumulation Steps": self.accelerator.gradient_accumulation_steps,
            "Total Optimization Steps": self.max_train_steps,
        }

        # run.config.update(hparam_dict)

    def prepare(self):
        """Just send all relevant objects through the accelerator to be placed on GPU."""
        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(self.model, self.optimizer)

    def train(self):
        # Sanity check the validation loop and sampling before training
        for epoch in range(self.first_epoch, self.max_epochs):
            # print(epoch)
            epoch_losses = []  # ← RIGHT HERE (line 196 in your file)

            for step, (batch, cond) in enumerate(self.train_loader.generate()):
                # print(step)
                # print(len(batch),batch[0].shape,batch[1].shape)
                self.model.train()
                # Skip steps until we reach the resumed step
                if (
                        self.load_path
                        and epoch == self.first_epoch
                        and step < self.resume_step
                ):
                    continue

                batch = batch.to(self.weight_dtype)
                with self.accelerator.accumulate(self.model):
                    loss = self.model_forward_pass(batch, cond)
                    epoch_losses.append(loss.item())

                self.ema_model.update()

                # Gather our losses for logging
                logs = {
                    "Training/Loss": loss.detach().item(),
                    "Learning Rate": self.optimizer.param_groups[0]["lr"],
                    # "Pixel-wise error":pixel_loss.detach().item(),
                    # "Masked loss":masked_loss.detach().item(),
                }
                # Log the loss to WANDB
                self.accelerator.log(logs, step=self.global_step)

                # Increment the global step (counts true iterations)
                if self.accelerator.sync_gradients:
                    self.global_step += 1

            # Calculate average epoch loss
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

            # Update learning rate scheduler
            self.lr_scheduler.step(avg_epoch_loss)

            # Check if this is the best epoch
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.best_epoch = epoch
                if self.accelerator.is_main_process:
                    print(f"New best model at epoch {epoch} with loss {avg_epoch_loss:.6f}")
                    self.save_best(epoch)

            # Save regular checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save(epoch)

            # If an error happened here, then the batch was empty leading to
            # division by zero
            if (epoch + 1) % self.sample_every == 0:
                pass
                # self.validation_loop()
                # self.sample()

        print(f"Training completed! Best model was at epoch {self.best_epoch} with loss {self.best_loss:.6f}")

    def get_original_sample(self, x_t, noise_pred, t):
        # Extract alpha values
        alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        # Rearrange the formula: x_0 = (x_t - sqrt(1-alpha_t)*eps) / sqrt(alpha_t)
        return (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

    def model_forward_pass(self, batch: torch.Tensor, cond: torch.Tensor):
        """Runs a single forward pass of the model and calculates loss.

        Args:
            batch: A batch of training samples.
        """
        # Sample a random timestep for each sample in the batch
        # sample noise that will be added to the samples
        noise = torch.randn_like(batch.to(self.device))

        bsz = batch.shape[0]

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (bsz,),
            device=batch.device,
        ).long()
        clean_samples = batch
        cond_map = cond.to(self.weight_dtype)

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_samples = self.scheduler.add_noise(clean_samples, noise, timesteps)

        with self.accelerator.accumulate(self.model):
            model_output = self.model(
                noisy_samples,
                timesteps,
                cond_map=cond_map,
            )

            # Make sure to get the right target for the loss
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(clean_samples, noise, timesteps)
            else:
                raise NotImplementedError("Only epsilon and v_prediction supported")

            # Calculate loss and update gradients
            mse_loss = calc_mse_loss(model_output, target, self.train_set.lats)
            # Calculate the avg conditional loss
            if hasattr(self.scheduler, "alphas_cumprod"):
                pred_original_sample = self.get_original_sample(noisy_samples, model_output, timesteps)
            elif self.scheduler.config.prediction_type == "v_prediction":
                pred_original_sample = self.scheduler.predict_start_from_v(noisy_samples, timesteps, model_output)
            else:
                pred_original_sample = self.scheduler.predict_start_from_noise(noisy_samples, timesteps, model_output)

            # Get the mean of both the clean and the predicted original sample
            clean_mean = clean_samples.mean(dim=-3)
            pred_mean = pred_original_sample.mean(dim=-3)
            cond_loss = ((clean_mean - pred_mean) ** 2).mean()

            # Calculate the loss
            loss = mse_loss + cond_loss * self.cond_loss_scaling

            # Scale the loss by cosine-weighted latitude
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    @torch.inference_mode()
    def validation_loop(self, sanity_check=False) -> None:
        """Runs a single epoch of validation.

        Updates the loss, logs it, and backpropagates the error.
        """
        self.model.eval()
        val_loss = 0

        for batch_idx, batch in enumerate(self.val_loader.generate()):
            # If we are sanity checking, only run 10 batches
            if sanity_check and batch_idx > 10:
                return

            val_loss += self.model_forward_pass(batch)[0].item()

        # Log the average
        self.accelerator.log(
            {"Validation/Loss": val_loss / len(self.val_loader)}, step=self.global_step
        )

    @torch.inference_mode()
    def sample(self) -> None:
        """Samples a batch of images from the model."""

        self.ema_model.eval()
        # Grab a random sample from validation set
        batch = random.choice(self.val_set).unsqueeze(0).to(self.accelerator.device)

        clean_samples = batch.to(self.weight_dtype)

        # Generate the samples
        gen_sample = generate_samples(
            clean_samples, self.scheduler, self.sample_steps, self.ema_model
        )

        # Turn the samples into xr datasets
        gen_ds = self.val_set.convert_tensor_to_xarray(gen_sample[0])
        val_ds = self.val_set.convert_tensor_to_xarray(clean_samples[0])

        # Create a gif of the samples
        gen_frames = create_gif(gen_ds)
        val_frames = create_gif(val_ds)

        # Log the gif to wandb
        for var, gif in gen_frames.items():
            self.accelerator.log(
                {f"Generated {var}": wandb.Video(gif, fps=4)}, step=self.global_step
            )

        for var, gif in val_frames.items():
            self.accelerator.log(
                {f"Original {var}": wandb.Video(gif, fps=4)}, step=self.global_step
            )

    def save_best(self, epoch: int):
        """Saves the best model checkpoint."""
        if self.save_name is None:
            return

        state_dict = {
            "EMA": self.ema_model.ema_model.state_dict(),
            "Unet": self.accelerator.unwrap_model(self.model).state_dict(),
            "Optimizer": self.optimizer.state_dict(),
            "Global Step": self.global_step,
            "LR_Scheduler": self.lr_scheduler.state_dict(),
            "Best Loss": self.best_loss,
            "Best Epoch": epoch,
        }

        # If the directory doesn't exist already create it
        os.makedirs(self.save_dir, exist_ok=True)

        # Create the save filename for best model
        base = self.save_name.split(".pt")[0]
        best_save_name = f"{base}_best.pt"
        best_save_path = os.path.join(self.save_dir, best_save_name)

        # Save the best model
        torch.save(state_dict, best_save_path, _use_new_zipfile_serialization=False)
        print(f"[INFO] Saved best model at epoch {epoch} with loss {self.best_loss:.6f}")

    def save(self, epoch: int):
        """Saves the state of training to disk."""
        if self.save_name is None:
            return
        else:
            state_dict = {
                "EMA": self.ema_model.ema_model.state_dict(),
                "Unet": self.accelerator.unwrap_model(self.model).state_dict(),
                "Optimizer": self.optimizer.state_dict(),
                "Global Step": self.global_step,
                "LR_Scheduler": self.lr_scheduler.state_dict(),
                "Best Loss": self.best_loss,
                "Best Epoch": self.best_epoch,
            }

            # If the directory doesn't exist already create it
            os.makedirs(self.save_dir, exist_ok=True)

            # Create the save filename and add the epoch number
            save_name = self.save_name.split(".pt")[0] + f"_{epoch}.pt"

            # Save the State dictionary to disk
            torch.save(state_dict, os.path.join(self.save_dir, save_name), _use_new_zipfile_serialization=False)

            base = self.save_name.split(".pt")[0]
            save_name = f"{base}_{epoch}.pt"
            save_path = os.path.join(self.save_dir, save_name)

            all_ckpts = [
                os.path.join(self.save_dir, f)
                for f in os.listdir(self.save_dir)
                if f.startswith(base + "_") and f.endswith(".pt") and "_best.pt" not in f
            ]

            # Sort by epoch number extracted from filename
            def extract_epoch(fname):
                try:
                    return int(fname.split("_")[-1].split(".")[0])
                except ValueError:
                    return -1  # fallback, should not happen

            all_ckpts_sorted = sorted(all_ckpts, key=extract_epoch, reverse=True)

            # Keep last 5, delete the rest
            keep_last = 5
            for ckpt in all_ckpts_sorted[keep_last:]:
                try:
                    os.remove(ckpt)
                except OSError:
                    pass

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Restore model
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["Unet"], strict=True)

        # Restore EMA (optional)
        if "EMA" in checkpoint and checkpoint["EMA"] is not None and hasattr(self, "ema_model"):
            try:
                # self.ema_model.load_state_dict(checkpoint["EMA"], strict=False)
                # self.ema_model = checkpoint["EMA"].to(self.device)

                ema_model_sd = checkpoint["EMA"]  # full EMA state dict (online_model + ema_model)

                # Extract only EMA weights and strip "ema_model." prefix
                # ema_model_sd = {
                #     k.replace("ema_model.", ""): v
                #     for k, v in ema_wrapped_sd.items()
                #     if k.startswith("ema_model.")
                # }

                ema_model = EMA(
                    self.model,
                    beta=0.9999,  # exponential moving average factor
                    update_after_step=100,  # only after this number of .update() calls will it start updating
                    update_every=10,
                ).to(self.device)
                ema_model.ema_model.load_state_dict(ema_model_sd)
                ema_model.eval()

            except Exception as e:
                print(f"[WARN] Could not load EMA: {e}")

        # Restore optimizer (optional)
        if "Optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["Optimizer"])
            except Exception as e:
                print(f"[WARN] Could not load optimizer state: {e}")

        # Restore learning rate scheduler
        if "LR_Scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["LR_Scheduler"])

        # Restore best model tracking
        if "Best Loss" in checkpoint:
            self.best_loss = checkpoint["Best Loss"]
        if "Best Epoch" in checkpoint:
            self.best_epoch = checkpoint["Best Epoch"]

        self.global_step = checkpoint.get("Global Step", 0)
        print(self.global_step, self.accelerator.gradient_accumulation_steps)
        self.resume_global_step = (
                self.global_step * self.accelerator.gradient_accumulation_steps
        )

        self.resume_step = self.resume_global_step % (
                self.num_steps_per_epoch * self.accelerator.gradient_accumulation_steps
        )

        self.first_epoch = self.global_step // self.num_steps_per_epoch

        print(f"[INFO] Loaded checkpoint from {path} (step {self.global_step})")
        if hasattr(self, 'best_epoch') and self.best_epoch >= 0:
            print(f"[INFO] Best model so far: epoch {self.best_epoch} with loss {self.best_loss:.6f}")