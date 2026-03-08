import torch
import pytest

pytest.importorskip('spikingjelly')

from models import LightSpikingTransformer, SNN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test_model_params():
    model_transformer = LightSpikingTransformer()
    num_params_transformer = count_parameters(model_transformer)

    model_snn = SNN()
    num_params_snn = count_parameters(model_snn)

    assert 0.11e6 <= num_params_transformer <= 0.13e6
    assert abs(num_params_transformer - num_params_snn) / num_params_snn < 0.1

def test_model_forward():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightSpikingTransformer()
    model.to(device)
    model.eval()
    
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 28, 28).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, 8)

if __name__ == "__main__":
    test_model_params()
    test_model_forward()
