import struct
import pytest
from app.services import redis_client as rc



def test_vector_serialisation_roundtrip():
    vec = [0.1, 0.2, 0.3, 0.4]
    blob = rc._ser(vec)
    out = list(struct.unpack(f"{len(blob)//4}f", blob))
    assert out == pytest.approx(vec)
