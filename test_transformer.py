import torch
from models import LightSpikingTransformer, SNN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def test_model_params():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== 测试LightSpikingTransformer模型参数量 ===")
    model_transformer = LightSpikingTransformer()
    model_transformer.to(device)
    num_params_transformer = count_parameters(model_transformer)
    print(f"LightSpikingTransformer 参数量: {num_params_transformer:,} ({num_params_transformer/1e6:.3f}M)")
    
    print("\n=== 对照SNN模型参数量 ===")
    model_snn = SNN()
    model_snn.to(device)
    num_params_snn = count_parameters(model_snn)
    print(f"SNN 参数量: {num_params_snn:,} ({num_params_snn/1e6:.3f}M)")
    
    print(f"\n参数量差异: {abs(num_params_transformer - num_params_snn):,} ({abs(num_params_transformer - num_params_snn)/num_params_snn*100:.2f}%)")
    
    if 0.11e6 <= num_params_transformer <= 0.13e6:
        print("\n✅ LightSpikingTransformer 参数量符合要求 (0.11M~0.13M)")
    else:
        print("\n❌ LightSpikingTransformer 参数量不符合要求")

def test_model_forward():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n=== 测试LightSpikingTransformer前向传播 ===")
    model = LightSpikingTransformer()
    model.to(device)
    model.eval()
    
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 28, 28).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    if output.shape == (batch_size, 8):
        print("✅ 前向传播测试通过!")
    else:
        print("❌ 前向传播测试失败!")

if __name__ == "__main__":
    test_model_params()
    test_model_forward()
