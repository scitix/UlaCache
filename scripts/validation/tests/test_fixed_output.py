"""Fixed histories select the actual next token at each native decode position."""
from types import SimpleNamespace
import pytest

from ulacache.tools.fixed_output import processor_string


class NativeProcessor:
    @classmethod
    def to_str(cls):
        return cls()


def test_fixed_histories_follow_request_state_and_explicit_positions():
    torch = pytest.importorskip('torch')
    processor = processor_string(NativeProcessor)
    logits = torch.randn(3, 13)
    params = [dict(replay_ids=[4, 7, 8], __req__=SimpleNamespace(output_ids=[])),
              dict(replay_ids=[6, 3, 2], __req__=SimpleNamespace(output_ids=[6])),
              dict(replay_ids=[1, 9, 5], replay_position=2)]
    processor(logits, params)
    assert logits.argmax(1).tolist() == [4, 3, 5]
    assert torch.isfinite(logits).sum().item() == 3


def test_fixed_sampler_rejects_incomplete_histories_and_vocabulary_errors():
    torch = pytest.importorskip('torch')
    processor = processor_string(NativeProcessor)
    with pytest.raises(ValueError, match='position'):
        processor(torch.zeros(1, 11), [dict(replay_ids=[1], replay_position=1)])
    with pytest.raises(ValueError, match='vocabulary'):
        processor(torch.zeros(1, 11), [dict(replay_ids=[11], replay_position=0)])
    with pytest.raises(ValueError, match='batch'):
        processor(torch.zeros(2, 11), [dict(replay_ids=[1], replay_position=0)])
