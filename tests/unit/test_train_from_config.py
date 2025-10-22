"""
Unit tests for train_from_config.py entry point.

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


class TestSetupDistributed:
    """Test suite for setup_distributed function."""

    def test_setup_distributed_deepspeed_path(self):
        """Test that setup_distributed returns valid tuple when using DeepSpeed."""
        from src.training.train_from_config import setup_distributed
        from argparse import Namespace

        # Create mock args with deepspeed enabled
        args = Namespace(deepspeed=True, local_rank=2)

        # Mock environment variables
        test_env = {
            "LOCAL_RANK": "3",  # Should override args.local_rank
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

                            # Verify return values are valid (not None or undefined)
                            assert rank is not None
                            assert world_size is not None
                            assert local_rank is not None

                            # Verify local_rank came from environment
                            assert local_rank == 3

                            # Verify distributed values
                            assert rank == 1
                            assert world_size == 4

    def test_setup_distributed_deepspeed_no_env(self):
        """Test DeepSpeed path when LOCAL_RANK not in environment."""
        from src.training.train_from_config import setup_distributed
        from argparse import Namespace

        args = Namespace(deepspeed=True, local_rank=5)

        # Clear LOCAL_RANK from environment
        test_env = os.environ.copy()
        test_env.pop("LOCAL_RANK", None)
        test_env.pop("RANK", None)
        test_env.pop("WORLD_SIZE", None)

        with patch.dict(os.environ, test_env, clear=True):
            with patch('deepspeed.init_distributed'):
                with patch('torch.distributed.is_initialized', return_value=True):
                    with patch('torch.distributed.get_rank', return_value=0):
                        with patch('torch.distributed.get_world_size', return_value=1):
                            rank, world_size, local_rank = setup_distributed(args)

                            # Should fall back to args.local_rank
                            assert local_rank == 5
                            assert rank == 0
                            assert world_size == 1

    def test_setup_distributed_deepspeed_import_error(self):
        """Test DeepSpeed path when deepspeed not installed."""
        from src.training.train_from_config import setup_distributed
        from argparse import Namespace

        args = Namespace(deepspeed=True, local_rank=-1)

        test_env = {"LOCAL_RANK": "2", "RANK": "0", "WORLD_SIZE": "1"}

        with patch.dict(os.environ, test_env, clear=False):
            # Mock deepspeed import failure
            with patch('builtins.__import__', side_effect=ImportError("No module named 'deepspeed'")):
                with patch('torch.distributed.init_process_group'):
                    with patch('torch.distributed.is_initialized', return_value=True):
                        with patch('torch.distributed.get_rank', return_value=0):
                            with patch('torch.distributed.get_world_size', return_value=1):
                                rank, world_size, local_rank = setup_distributed(args)

                                # Should still return valid values
                                assert rank is not None
                                assert world_size is not None
                                assert local_rank is not None

    def test_setup_distributed_manual_distributed(self):
        """Test manual distributed setup (not using DeepSpeed)."""
        from src.training.train_from_config import setup_distributed
        from argparse import Namespace

        args = Namespace(deepspeed=False, local_rank=-1)

        test_env = {
            "RANK": "2",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "6",
        }

        with patch.dict(os.environ, test_env, clear=False):
            with patch('torch.cuda.set_device'):
                with patch('torch.distributed.init_process_group'):
                    with patch('torch.distributed.is_initialized', return_value=True):
                        with patch('torch.distributed.get_rank', return_value=2):
                            with patch('torch.distributed.get_world_size', return_value=8):
                                rank, world_size, local_rank = setup_distributed(args)

                                assert rank == 2
                                assert world_size == 8
                                assert local_rank == 6

    def test_setup_distributed_single_gpu_fallback(self):
        """Test fallback to single-GPU mode when no distributed env vars."""
        from src.training.train_from_config import setup_distributed
        from argparse import Namespace

        args = Namespace(deepspeed=False, local_rank=-1)

        # Clear all distributed environment variables
        test_env = os.environ.copy()
        test_env.pop("RANK", None)
        test_env.pop("WORLD_SIZE", None)
        test_env.pop("LOCAL_RANK", None)

        with patch.dict(os.environ, test_env, clear=True):
            with patch('torch.distributed.is_initialized', return_value=False):
                rank, world_size, local_rank = setup_distributed(args)

                # Should fall back to single-GPU defaults
                assert rank == 0
                assert world_size == 1
                assert local_rank == 0
