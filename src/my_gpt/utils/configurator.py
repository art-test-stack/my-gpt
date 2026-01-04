
def compute_chinchilla_nb_tokens(model_params: int, dataset_size_tokens: int) -> float:
    "https://arxiv.org/abs/2203.15556"
    optimal_tokens = 20 * (model_params ** 0.75)
    return min(optimal_tokens, dataset_size_tokens)

def compute_chinchilla_ratio(model_params: int, dataset_size_tokens: int) -> float:
    optimal_tokens = compute_chinchilla_nb_tokens(model_params, dataset_size_tokens)
    return optimal_tokens / model_params

def compute_model_flops(model_params: int, nb_tokens: int) -> float:
    "https://arxiv.org/abs/2203.15556"
    return 6 * model_params * nb_tokens / 1e12  # in TFLOPS