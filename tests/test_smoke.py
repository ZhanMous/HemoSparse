import pytest
import torch

pytest.importorskip('spikingjelly')

from models import ANN, SNN, DenseSNN


def test_ann_forward_shape():
    model = ANN()
    model.eval()
    x = torch.randn(8, 3, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 8)

def test_snn_dense_forward_shape():
    snn = SNN(T=2)
    dsn = DenseSNN(T=2)
    snn.eval()
    dsn.eval()
    x = torch.randn(8, 3, 28, 28)
    with torch.no_grad():
        out1 = snn(x)
        out2 = dsn(x)
    assert out1.shape == (8, 8)
    assert out2.shape == (8, 8)
