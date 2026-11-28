from . import console


class ModelParameterCounter:
    """
    A utility class to count the number of parameters in a PyTorch model.
    """
    
    @staticmethod
    def count_parameters(model):
        """
        Count total and trainable parameters in a PyTorch model.

        Args:
            model (nn.Module): The model to analyze.

        Returns:
            total_params (int): Total number of parameters.
            trainable_params (int): Number of trainable (requires_grad=True) parameters.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        def format_p(n):
            if n >= 1e9: return f"{n/1e9:.3f}B"
            if n >= 1e6: return f"{n/1e6:.3f}M"
            return f"{n:,}"

        console.info(
            f"Total parameters: {format_p(total_params)} ({total_params:,}), "
            f"Trainable parameters: {format_p(trainable_params)} ({trainable_params:,})"
        )

        return total_params, trainable_params