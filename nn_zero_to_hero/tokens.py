from typing import Dict, List, Tuple, TypeVar

import torch
import torch.nn.functional as F

T = TypeVar("T")


def tokens_to_int_mapping(tokens: List[T]) -> Tuple[Dict[T, int], Dict[int, T]]:
    stoi = {s: i + 1 for i, s in enumerate(tokens)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def sample_from_model(
    model: torch.nn.Module,
    *,
    block_size: int,
    device: torch.device,
    itos: Dict[int, str],
    generator: torch.Generator
) -> str:
    model.eval()
    out = []
    context = [0] * block_size  # initialize with all ...
    while True:
        with torch.no_grad():
            logits = model.forward(torch.tensor([context], device=device))
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    return "".join(itos[i] for i in out)
