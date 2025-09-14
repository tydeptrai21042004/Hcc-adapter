
import torch

def _num_bytes_of_params(model, trainable_bits=32, frozen_bits=8):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable * (trainable_bits/8), frozen * (frozen_bits/8)

@torch.no_grad()
def profile_memory_cost(model, input_shape, has_classifier=True,
                        activation_bits=32, trainable_param_bits=32, frozen_param_bits=8,
                        batch_size=8):
    device = next(model.parameters()).device
    x = torch.zeros((batch_size,)+tuple(input_shape[1:]), device=device)
    model.eval()

    act_bytes = 0
    # Only track GPU peak mem if the model is on CUDA
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(None)  # reset for current device
        _ = model(x)
        act_bytes = torch.cuda.max_memory_allocated(None)

    trainable_b, frozen_b = _num_bytes_of_params(model, trainable_param_bits, frozen_param_bits)
    param_bytes = trainable_b + frozen_b
    mem_cost = param_bytes + act_bytes
    details = {"param_size": param_bytes, "act_size": act_bytes}
    return mem_cost, details
