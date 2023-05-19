#
import torch

msg = f"pytorch version: {torch.__version__} \n" \
      f"pytorch CUDA version: {torch.version.cuda} \n" \
      f"pytorch CUDA available: {torch.cuda.is_available()} \n" \
      f"--------------------------------------- \n\n"
print(msg)
