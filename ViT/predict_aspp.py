"""Simple prediction runner used by `predict_swin_unet.py`.

Provides `run_prediction(model, dataloader, device, color_map, save_dir, max_save)`.
This is a lightweight local implementation intended for quick qualitative outputs.
"""
from pathlib import Path
import os
import cv2
import numpy as np
import torch


def _apply_color_map(pred_mask: np.ndarray, color_map) -> np.ndarray:
    # pred_mask: H x W (class indices)
    palette = np.array(color_map, dtype=np.uint8)
    h, w = pred_mask.shape
    colored = palette[pred_mask.reshape(-1)].reshape(h, w, 3)
    return colored


def run_prediction(model, dataloader, device, color_map, save_dir: Path, max_save=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    alpha = 0.6
    with torch.no_grad():
        for batch in dataloader:
            # Expect dataset to yield (tensor, orig_bgr, filename)
            # If batch_size==1, dataloader will return tensors with added batch dim
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                imgs, origs, names = batch
            else:
                # fallback: assume dataset yields (img, orig, name) per sample
                imgs, origs, names = batch

            # handle batch or single sample
            if isinstance(names, (list, tuple)):
                batch_size = len(names)
            else:
                batch_size = 1

            # Ensure imgs is tensor on device
            imgs = imgs.to(device)
            outputs = model(imgs)
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs

            # logits: (N, C, H, W)
            probs = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(probs.shape[0]):
                mask = probs[i]
                # origs may be list of arrays or a single array
                orig = origs[i] if isinstance(origs, (list, tuple)) or getattr(origs, 'shape', None) and origs.shape[0] > 1 else origs
                if isinstance(orig, torch.Tensor):
                    orig = orig.cpu().numpy()
                # orig expected BGR uint8
                base = Path(names[i]).stem if isinstance(names, (list, tuple)) else Path(names).stem

                color = _apply_color_map(mask, color_map)
                # color is RGB, convert to BGR for cv2
                color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

                out_mask_color = save_dir / f"{base}_mask_color.png"
                cv2.imwrite(str(out_mask_color), color_bgr)

                # overlay
                if isinstance(orig, np.ndarray):
                    overlay = cv2.addWeighted(orig, 1.0 - alpha, color_bgr, alpha, 0)
                    out_overlay = save_dir / f"{base}_overlay.png"
                    cv2.imwrite(str(out_overlay), overlay)

                saved += 1
                if max_save and saved >= max_save:
                    return