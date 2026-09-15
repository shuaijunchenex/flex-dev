import torch

class DeviceChecker:
    """
    Utility class for detecting and providing the best available hardware device (CPU, CUDA, or MPS).
    """

    @staticmethod
    def get_device() -> torch.device:
        """
        Returns the best available torch.device.
        Priority: CUDA > MPS (Apple Silicon) > CPU
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        # Check for Apple Silicon MPS support
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Note: Some older torch versions might have the attribute but not be functional
            try:
                # Test if mps is actually usable
                torch.zeros(1).to("mps")
                return torch.device("mps")
            except Exception:
                pass
                
        return torch.device("cpu")

    @staticmethod
    def get_device_type() -> str:
        """
        Returns the string name of the best available device ('cuda', 'mps', or 'cpu').
        """
        return DeviceChecker.get_device().type

    @staticmethod
    def is_cuda_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def is_mps_available() -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
