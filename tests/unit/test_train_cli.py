"""
Unit tests for train.py (legacy CLI entry point).

Tests the setup_distributed function to ensure proper handling of DeepSpeed
and torch.distributed initialization paths.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLegacySetupDistributed:
    """Test suite for legacy train.py setup_distributed function."""

    def test_setup_distributed_deepspeed_path(self):
        """Test that setup_distributed returns valid tuple when using DeepSpeed."""
        from src.training.train import setup_distributed
        from argparse import Namespace

        # Create mock args with deepspeed enabled
        args = Namespace(deepspeed=True, local_rank=2)

        # Mock environment variables
        test_env = {
            "LOCAL_RANK": "3",
            "RANK": "1",
            "WORLD_SIZE": "4",
        }

        with patch.dict(os.environ, test_env, clear=False):
            # Mock deepspeed.init_distributed
            with patch('deepspeed.init_distributed') as mock_ds_init:
                # Mock torch.distributed
                with patch('torch.distributed.is_initialized', return_value=True):
                    with patch('torch.distributed.get_rank', return_value=1):
                        with patch('torch.distributed.get_world_size', return_value=4):
                            rank, world_size, local_rank = setup_distributed(args)

                            # Verify DeepSpeed was initialized
                            mock_ds_init.assert_called_once()

                            # Verify return values are valid
                            assert rank is not None
                            assert world_size is not None
                            assert local_rank is not None

                            # Verify local_rank came from environment
                            assert local_rank == 3

                            # Verify distributed values
                            assert rank == 1
                            assert world_size == 4

    def test_setup_distributed_deepspeed_fallback_to_args(self):
        """Test DeepSpeed path falls back to args.local_rank when no env var."""
        from src.training.train import setup_distributed
        from argparse import Namespace

        args = Namespace(deepspeed=True, local_rank=7)

        test_env = os.environ.copy()
        test_env.pop("LOCAL_RANK", None)

        with patch.dict(os.environ, test_env, clear=True):
            with patch('deepspeed.init_distributed'):
                with patch('torch.distributed.is_initialized', return_value=True):
                    with patch('torch.distributed.get_rank', return_value=0):
                        with patch('torch.distributed.get_world_size', return_value=1):
                            rank, world_size, local_rank = setup_distributed(args)

                            # Should use args.local_rank
                            assert local_rank == 7

    def test_setup_distributed_manual_mode(self):
        """Test manual distributed setup (not using DeepSpeed)."""
        from src.training.train import setup_distributed
        from argparse import Namespace

        args = Namespace(deepspeed=False, local_rank=-1)

        test_env = {
            "RANK": "3",
            "WORLD_SIZE": "16",
            "LOCAL_RANK": "11",
        }

        with patch.dict(os.environ, test_env, clear=False):
            with patch('torch.cuda.set_device'):
                with patch('torch.distributed.init_process_group'):
                    with patch('torch.distributed.is_initialized', return_value=True):
                        with patch('torch.distributed.get_rank', return_value=3):
                            with patch('torch.distributed.get_world_size', return_value=16):
                                rank, world_size, local_rank = setup_distributed(args)

                                assert rank == 3
                                assert world_size == 16
                                assert local_rank == 11

    def test_setup_distributed_single_gpu(self):
        """Test fallback to single-GPU when no distributed env vars."""
        from src.training.train import setup_distributed
        from argparse import Namespace

        args = Namespace(deepspeed=False, local_rank=-1)

        test_env = os.environ.copy()
        test_env.pop("RANK", None)
        test_env.pop("WORLD_SIZE", None)
        test_env.pop("LOCAL_RANK", None)

        with patch.dict(os.environ, test_env, clear=True):
            with patch('torch.distributed.is_initialized', return_value=False):
                rank, world_size, local_rank = setup_distributed(args)

                # Should fall back to defaults
                assert rank == 0
                assert world_size == 1
                assert local_rank == 0
