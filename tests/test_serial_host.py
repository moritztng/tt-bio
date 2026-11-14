"""runtime.serial_host runs its block on one torch thread and hands the pool back after."""
import torch

from tt_bio import runtime


def test_one_thread_inside_and_restored_after(monkeypatch):
    torch.set_num_threads(4)
    with runtime.serial_host():
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == 4


def test_restored_when_the_block_raises():
    torch.set_num_threads(4)
    try:
        with runtime.serial_host():
            raise RuntimeError
    except RuntimeError:
        pass
    assert torch.get_num_threads() == 4


def test_the_switch_leaves_the_pool_alone(monkeypatch):
    monkeypatch.setattr(runtime, "SERIAL_HOST", False)
    torch.set_num_threads(4)
    with runtime.serial_host():
        assert torch.get_num_threads() == 4
