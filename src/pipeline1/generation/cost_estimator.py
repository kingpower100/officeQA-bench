def estimate_cost(input_tokens: int, output_tokens: int, input_per_1k_tokens_usd: float, output_per_1k_tokens_usd: float) -> float:
    return (input_tokens / 1000.0) * input_per_1k_tokens_usd + (output_tokens / 1000.0) * output_per_1k_tokens_usd
