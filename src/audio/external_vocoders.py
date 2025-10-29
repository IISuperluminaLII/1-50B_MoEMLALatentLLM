"""
External Vocoder Wrappers.

Provides wrappers for external high-quality vocoders:
- HiFiGAN (https://github.com/jik876/hifi-gan)
- Vocos (https://github.com/charactr-platform/vocos)
- BigVGAN-v2 (https://github.com/NVIDIA/BigVGAN)

To use these vocoders:
1. Install the respective library
2. Download pretrained checkpoints
3. Initialize the wrapper with checkpoint path

For quick prototyping, use iSTFTNetVocoder instead.
"""

import torch
from typing import Optional
from .base_vocoder import BaseVocoder


class HiFiGANVocoder(BaseVocoder):
    """
    Wrapper for HiFiGAN vocoder.

    Installation:
        pip install hifigan

    Usage:
        vocoder = HiFiGANVocoder(checkpoint_path="path/to/checkpoint")
        waveform = vocoder.synthesize(mel_spectrogram)

    Args:
        checkpoint_path: Path to HiFiGAN checkpoint
        config_path: Path to config file (optional)
        sample_rate: Audio sample rate
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)

        try:
            # Attempt to import HiFiGAN
            from hifigan.models import Generator
            from hifigan.env import AttrDict
            import json

            # Load config
            if config_path:
                with open(config_path) as f:
                    config = json.load(f)
                h = AttrDict(config)
            else:
                # Use default config
                h = self._get_default_config()

            # Load generator
            self.generator = Generator(h).to('cpu')

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.generator.load_state_dict(checkpoint['generator'])
            self.generator.eval()
            self.generator.remove_weight_norm()

            self.hop_length = h.hop_size
            print(f"HiFiGAN loaded from: {checkpoint_path}")

        except ImportError:
            raise ImportError(
                "HiFiGAN not installed. Install with: pip install hifigan\n"
                "Or use iSTFTNetVocoder for baseline quality."
            )

    def _get_default_config(self):
        """Default HiFiGAN config for 16kHz."""
        from hifigan.env import AttrDict
        return AttrDict({
            "resblock": "1",
            "num_gpus": 0,
            "batch_size": 1,
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.999,
            "seed": 1234,
            "upsample_rates": [8, 8, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "segment_size": 8192,
            "num_mels": 80,
            "num_freq": 1025,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 16000,
        })

    def synthesize(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Synthesize waveform from mel-spectrogram.

        Args:
            mel_spectrogram: [batch, n_mels, time] mel-spectrogram

        Returns:
            waveform: [batch, samples] synthesized audio
        """
        with torch.no_grad():
            waveform = self.generator(mel_spectrogram)
            waveform = waveform.squeeze(1)  # Remove channel dimension
        return waveform

    def get_output_length(self, feature_length: int) -> int:
        """Calculate output length."""
        return feature_length * self.hop_length


class VocosVocoder(BaseVocoder):
    """
    Wrapper for Vocos vocoder.

    Installation:
        pip install vocos

    Usage:
        vocoder = VocosVocoder(checkpoint_path="path/to/checkpoint")
        waveform = vocoder.synthesize(features)

    Args:
        checkpoint_path: Path to Vocos checkpoint or model name
        sample_rate: Audio sample rate
    """

    def __init__(
        self,
        checkpoint_path: str = "charactr/vocos-mel-24khz",  # Default pretrained
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)

        try:
            from vocos import Vocos

            # Load Vocos model
            self.model = Vocos.from_pretrained(checkpoint_path)
            self.model.eval()

            print(f"Vocos loaded: {checkpoint_path}")

        except ImportError:
            raise ImportError(
                "Vocos not installed. Install with: pip install vocos\n"
                "Or use iSTFTNetVocoder for baseline quality."
            )

    def synthesize(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Synthesize waveform from features.

        Args:
            features: [batch, n_mels, time] or [batch, freq, time]

        Returns:
            waveform: [batch, samples]
        """
        with torch.no_grad():
            waveform = self.model.decode(features)
        return waveform

    def get_output_length(self, feature_length: int) -> int:
        """Calculate output length."""
        # Vocos typically uses 256 hop length
        return feature_length * 256


class BigVGANVocoder(BaseVocoder):
    """
    Wrapper for BigVGAN-v2 vocoder.

    Installation:
        git clone https://github.com/NVIDIA/BigVGAN
        cd BigVGAN
        pip install -r requirements.txt

    Usage:
        vocoder = BigVGANVocoder(checkpoint_path="path/to/checkpoint")
        waveform = vocoder.synthesize(mel_spectrogram)

    Args:
        checkpoint_path: Path to BigVGAN checkpoint
        config_path: Path to config file
        sample_rate: Audio sample rate
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        sample_rate: int = 16000
    ):
        super().__init__(sample_rate=sample_rate)

        try:
            import json

            # Try importing BigVGAN directly first (if installed in environment)
            try:
                from bigvgan import BigVGAN
            except ImportError:
                # BigVGAN not in path, try using BIGVGAN_PATH env var
                import sys
                import os

                bigvgan_path = os.environ.get('BIGVGAN_PATH')
                if not bigvgan_path:
                    raise OSError(
                        "BigVGAN not found in Python path and BIGVGAN_PATH environment variable not set. "
                        "Either install BigVGAN or set BIGVGAN_PATH:\n"
                        "  export BIGVGAN_PATH=/path/to/BigVGAN"
                    )

                sys.path.append(bigvgan_path)
                from bigvgan import BigVGAN

            # Load config
            with open(config_path) as f:
                config = json.load(f)

            # Load model
            self.model = BigVGAN(**config['model'])
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['generator'])
            self.model.eval()
            self.model.remove_weight_norm()

            self.hop_length = config.get('hop_length', 256)
            print(f"BigVGAN loaded from: {checkpoint_path}")

        except ImportError:
            raise ImportError(
                "BigVGAN not installed. Clone from: https://github.com/NVIDIA/BigVGAN\n"
                "Or use iSTFTNetVocoder for baseline quality."
            )

    def synthesize(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Synthesize waveform from mel-spectrogram.

        Args:
            mel_spectrogram: [batch, n_mels, time]

        Returns:
            waveform: [batch, samples]
        """
        with torch.no_grad():
            waveform = self.model(mel_spectrogram)
            waveform = waveform.squeeze(1)
        return waveform

    def get_output_length(self, feature_length: int) -> int:
        """Calculate output length."""
        return feature_length * self.hop_length


def create_vocoder(
    vocoder_type: str = "istft",
    **kwargs
) -> BaseVocoder:
    """
    Factory function to create vocoder.

    Args:
        vocoder_type: "istft", "hifigan", "vocos", or "bigvgan"
        **kwargs: Vocoder-specific arguments

    Returns:
        BaseVocoder instance
    """
    from .istft_vocoder import iSTFTNetVocoder

    if vocoder_type == "istft":
        return iSTFTNetVocoder(**kwargs)
    elif vocoder_type == "hifigan":
        return HiFiGANVocoder(**kwargs)
    elif vocoder_type == "vocos":
        return VocosVocoder(**kwargs)
    elif vocoder_type == "bigvgan":
        return BigVGANVocoder(**kwargs)
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")


# Example usage
if __name__ == "__main__":
    print("External Vocoder Wrappers")
    print("=" * 70)
    print("\nAvailable vocoders:")
    print("  1. iSTFT (built-in, baseline quality)")
    print("  2. HiFiGAN (high quality, requires installation)")
    print("  3. Vocos (high quality, requires installation)")
    print("  4. BigVGAN-v2 (highest quality, requires manual setup)")
    print("\nFor prototyping, use iSTFTNetVocoder.")
    print("\nExample usage:")
    print("  from src.audio.external_vocoders import create_vocoder")
    print("  vocoder = create_vocoder('istft')")
    print("  waveform = vocoder.synthesize(stft_complex)")
