"""
FS Reference Predictor
ResNet18 backbone (pretrained on ImageNet) + 3-output regression head
입력: 이미지 (224×224×3)
출력: [predicted_lpips, predicted_clip_sim, predicted_arc_sim]
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FSReferencePredictor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # 1. Pretrained ResNet18 백본 로드
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            print('[FSPredictor] Loaded pretrained ResNet18 (ImageNet)')
        else:
            self.backbone = resnet18(weights=None)
            print('[FSPredictor] Loaded ResNet18 (random init)')
        
        # 2. 마지막 layer 교체: 1000개 클래스 분류 → 3개 수치 회귀
        in_features = self.backbone.fc.in_features   # 512
        self.backbone.fc = nn.Linear(in_features, 3)
    
    def forward(self, x):
        return self.backbone(x)


def count_parameters(model):
    """모델의 학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 테스트
    model = FSReferencePredictor(pretrained=True)
    
    # 파라미터 개수
    total = count_parameters(model)
    print(f'\nTotal trainable params: {total:,} ({total/1e6:.2f}M)')
    
    # 더미 입력으로 forward 테스트
    dummy = torch.randn(2, 3, 224, 224)   # batch=2 이미지
    output = model(dummy)
    print(f'\nInput shape:  {dummy.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Sample output: {output[0].detach().numpy()}')
    
    # GPU 사용 가능하면 GPU로
    if torch.cuda.is_available():
        model = model.cuda()
        dummy = dummy.cuda()
        output = model(dummy)
        print(f'\n[CUDA] Output (GPU): {output[0].detach().cpu().numpy()}')
        print(f'[CUDA] Device: {next(model.parameters()).device}')