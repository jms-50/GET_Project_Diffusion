# ==============================================================================
# DDPM & Classifier-Free Guidance (CFG) 라인별 주석이 포함된 전체 파이썬 라이브러리
# ==============================================================================

import math, random, os, numpy as np  # 1. 수학 연산(math), 난수 생성(random), OS 인터페이스(os), 수치 연산(numpy) 모듈 임포트
import torch, torch.nn as nn, torch.nn.functional as F  # 2. PyTorch 핵심 라이브러리, 신경망 모듈(nn), 기능적 함수(F) 임포트
from torch.utils.data import DataLoader  # 3. PyTorch 데이터 로더 모듈 임포트 (배치 처리 및 병렬 로딩)
from torchvision import datasets, transforms, utils as vutils  # 4. vision 관련 데이터셋, 이미지 변환(transforms), 유틸리티(vutils) 임포트
from einops import rearrange  # 5. 텐서 차원 재배열 라이브러리 (einops.rearrange) 임포트
from tqdm import tqdm  # 6. 진도율 시각화 라이브러리 tqdm 임포트

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 7. GPU(CUDA)가 이용 가능하면 'cuda', 사용 불가능하면 'cpu' 장치 설정
torch.manual_seed(0)  # 8. PyTorch CPU 및 GPU 난수 시드(seed)를 0으로 고정하여 실험 재현성 확보
np.random.seed(0)  # 9. NumPy 난수 시드를 0으로 고정
random.seed(0)  # 10. Python 기본 random 모듈 시드를 0으로 고정

transform = transforms.Compose([  # 11. MNIST 데이터 전처리 파이프라인 정의 (이미지 텐서 변환 및 [-1, 1] 범위를 위한 정규화)
    transforms.ToTensor(),  # 12. PIL 이미지나 numpy 배열을 [0.0, 1.0] 범위의 torch.FloatTensor (1, 28, 28)로 변환
    transforms.Normalize((0.5,), (0.5,))  # 13. 평균 0.5, 표준편차 0.5로 정규화하여 [0.0, 1.0] 범위의 픽셀값을 [-1.0, 1.0] 범위로 매핑
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)  # 14. MNIST 훈련 데이터셋 다운로드 및 전처리 적용 (root 디렉토리: ./data)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, drop_last=True)  # 15. DataLoader 생성: 배치 크기 64, 데이터 셔플 활성화, 2개 워커 프로세스 사용, 마지막 남은 짜투리 배치 버림
num_classes = 10  # 16. MNIST 데이터셋의 총 클래스 개수 설정 (숫자 0부터 9까지 총 10개)


# ==============================================================================
# Diffusion Schedule 클래스 정의
# ==============================================================================
class DiffusionSchedule:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02):  # 17. DiffusionSchedule 클래스 초기화 메서드 (총 타임스텝 T=200, beta 시작값 1e-4, 종료값 0.02)
        self.T = T  # 18. 총 타임스텝 수 T를 객체 변수에 저장
        betas = torch.linspace(beta_start, beta_end, T)  # 19. beta_start부터 beta_end까지 T개의 균등 분할된 선형 Beta 노이즈 스케줄 텐서 생성
        alphas = 1.0 - betas  # 20. alpha_t = 1.0 - beta_t 계산 (각 타임스텝별 보존 비율)
        alphas_bar = torch.cumprod(alphas, dim=0)  # 21. alpha_bar_t = cumprod(alpha) 계산 (타임스텝 t까지의 누적 곱, alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t)

        self.register(betas, alphas, alphas_bar)  # 22. register 메서드를 호출하여 계산된 계수들을 GPU/CPU 장치로 이동 및 텐서 변수로 등록

    def register(self, betas, alphas, alphas_bar):  # 23. 계산된 계수 텐서들을 지정된 장치(device)로 이동 및 뷰 변환용 속성 생성 메서드
        self.betas = betas.to(device)  # 24. betas 텐서를 디바이스로 이동
        self.alphas = alphas.to(device)  # 25. alphas 텐서를 디바이스로 이동
        self.alphas_bar = alphas_bar.to(device)  # 26. alphas_bar 텐서를 디바이스로 이동
        self.sqrt_alphas = torch.sqrt(self.alphas)  # 27. sqrt(alpha_t) 계수 계산 (역방향 과정 복원용)
        self.sqrt_one_minus_alphas = torch.sqrt(1.0 - self.alphas)  # 28. sqrt(1 - alpha_t) 계수 계산
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)  # 29. sqrt(alpha_bar_t) 계수 계산 (Forward Process x_t 생성용 원본 데이터 비율)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)  # 30. sqrt(1 - alpha_bar_t) 계수 계산 (Forward Process x_t 생성용 노이즈 비율)
        self.one_over_sqrt_alphas = 1.0 / self.sqrt_alphas  # 31. 1.0 / sqrt(alpha_t) 역수 계수 계산

    def sample_timesteps(self, bsz):  # 32. 훈련 시 무작위 타임스텝 t를 배치 크기(bsz)만큼 균일 분포 Uniform(0, T-1)에서 샘플링하는 메서드
        return torch.randint(0, self.T, (bsz,), device=device, dtype=torch.long)  # 33. [0, T-1] 범위에서 정수형 타임스텝 텐서 생성 (크기: (bsz,))

    def q_sample(self, x0, t, eps):  # 34. Forward Process: x0와 노이즈 eps로부터 타임스텝 t의 노이즈 이미지 x_t 생성 메서드
        s1 = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1)  # 35. x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps 공식 사용 36. sqrt_alphas_bar[t]를 추출 후 4차원 텐서 (B, 1, 1, 1)로 변환하여 브로드캐스팅 준비
        s2 = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)  # 37. sqrt_one_minus_alphas_bar[t]를 추출 후 4차원 텐서 (B, 1, 1, 1)로 변환
        return s1 * x0 + s2 * eps  # 38. 정방향 노이즈 이미지 x_t 리턴 (Shape: (B, 1, 28, 28))

    def posterior_mean_variance(self, xt, eps_pred, t):  # 39. Reverse Process: x_t와 예측된 노이즈 eps_pred로부터 posterior 평균(mu)과 분산(var) 산출 메서드
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)  # 40. t: (B,) 타임스텝 인덱스, xt: (B, 1, 28, 28) 이미지, eps_pred: (B, 1, 28, 28) 예측 노이즈 41. 해당 타임스텝의 alpha_t를 추출하여 (B, 1, 1, 1) 형태로 변환
        alpha_bar_t = self.alphas_bar[t].view(-1, 1, 1, 1)  # 42. 해당 타임스텝의 alpha_bar_t를 추출하여 (B, 1, 1, 1) 형태로 변환

        t_prev = torch.clamp(t - 1, min=0)  # 43. t-1 타임스텝 계산 (t가 0일 때는 클램프하여 t_prev=0 지정)
        alpha_bar_prev = self.alphas_bar[t_prev].view(-1, 1, 1, 1)  # 44. t-1 타임스텝의 alpha_bar_{t-1} 추출
        alpha_bar_prev = torch.where((t == 0).view(-1, 1, 1, 1), torch.ones_like(alpha_bar_prev), alpha_bar_prev)  # 45. t == 0인 지점에서는 alpha_bar_{-1} = 1.0으로 설정 (초기 조건 처리)

        beta_t = self.betas[t].view(-1, 1, 1, 1)  # 46. 해당 타임스텝의 beta_t 추출 (B, 1, 1, 1)
        beta_tilde_t = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t  # 47. DDPM Posterior Variance (beta_tilde_t) 계산 공식: ((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * beta_t

        mean = (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred) / torch.sqrt(alpha_t)  # 48. DDPM Posterior Mean (mu) 계산 공식: (1 / sqrt(alpha_t)) * (xt - ((1 - alpha_t) / sqrt(1 - alpha_bar_t)) * eps_pred)

        var = beta_tilde_t  # 49. Posterior 분산(var)을 beta_tilde_t로 설정
        return mean, var  # 50. 계산된 역방향 분포의 평균(mean)과 분산(var) 튜플 리턴


# ==============================================================================
# Denoising U-Net 신경망 구조 및 임베딩 모듈
# ==============================================================================

def sinusoidal_time_embedding(t, dim):  # 51. 타임스텝 t를 연속적인 Sinusoidal Vector로 변환하는 시분할 positional embedding 함수
    half = dim // 2  # 52. 임베딩 차원의 절반 크기 구함 (sin 및 cos에 각각 반씩 할당)
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)  # 53. 주파수 스케일 벡터 계산: exp(-log(10000) * [0..half-1] / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # 54. 타임스텝 t와 주파수 벡터의 외적(Outer Product) 수행 -> Shape: (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # 55. sin(args)와 cos(args)를 차원 1 방향으로 결합하여 텐서 생성 -> Shape: (B, dim)
    if dim % 2 == 1:  # 56. 만약 요청된 dim이 홀수일 경우 마지막 차원에 0 패딩 1칸 추가
        emb = F.pad(emb, (0, 1))
    return emb  # 57. 완성된 sinusoidal time embedding 텐서 반환 (Shape: (B, dim))


class AdaGN(nn.Module):  # 58. Adaptive Group Normalization (AdaGN) 모듈 클래스 정의
    def __init__(self, num_channels, cond_dim, groups=8):  # 59. AdaGN 초기화 메서드 (채널 수, 조건 벡터 차원, 그룹 수 8)
        super().__init__()  # 60. 상위 nn.Module 초기화
        self.gn = nn.GroupNorm(groups, num_channels, affine=False)  # 61. GroupNorm 생성 (아핀 변환 파라미터 affine=False 지정하여 기본 scale/shift 비활성화)
        self.fc = nn.Linear(cond_dim, num_channels * 2)  # 62. 조건 벡터 cond로부터 Scale(s) 및 Shift(b)를 동시에 예측하는 선형 레이어 생성 (출력 차원: num_channels * 2)
        nn.init.zeros_(self.fc.weight)  # 63. 선형 레이어 가중치(weight)를 0으로 초기화하여 초기 단계에서 항등 변환에 가깝게 설정
        nn.init.zeros_(self.fc.bias)  # 64. 선형 레이어 편향(bias)을 0으로 초기화

    def forward(self, x, cond):  # 65. AdaGN 순전파(forward) 메서드 (입력 특징 맵 x, 조건 벡터 cond)
        h = self.gn(x)  # 66. 입력 x에 그룹 정규화(Group Normalization) 적용 -> h: (B, C, H, W)
        s, b = self.fc(cond).chunk(2, dim=1)  # 67. 조건 벡터 cond를 선형 레이어에 통과시킨 후 chunk(2)하여 scale(s)과 shift(b)로 분할 -> 각각 (B, C)
        s = s.unsqueeze(-1).unsqueeze(-1)  # 68. scale 텐서를 spatial 차원 확장을 위해 4차원 (B, C, 1, 1)으로 변환
        b = b.unsqueeze(-1).unsqueeze(-1)  # 69. shift 텐서를 spatial 차원 확장을 위해 4차원 (B, C, 1, 1)으로 변환
        return h * (1 + s) + b  # 70. 적응형 아핀 변환 적용: h * (1 + s) + b 연산 후 반환


class ResBlock(nn.Module):  # 71. AdaGN 조건 결합을 포함하는 Residual Block 클래스 정의
    def __init__(self, c_in, c_out, cond_dim, dropout=0.0):  # 72. ResBlock 초기화 (입력 채널, 출력 채널, 조건 차원, 드롭아웃 비율)
        super().__init__()  # 73. 상위 모듈 초기화
        self.in_conv = nn.Conv2d(c_in, c_out, 3, padding=1)  # 74. 첫 번째 3x3 합성곱 레이어 (패딩 1로 공간 해상도 유지)
        self.ada1 = AdaGN(c_out, cond_dim)  # 75. 첫 번째 AdaGN 정규화 레이어
        self.mid_conv = nn.Conv2d(c_out, c_out, 3, padding=1)  # 76. 중간 3x3 합성곱 레이어
        self.ada2 = AdaGN(c_out, cond_dim)  # 77. 두 번째 AdaGN 정규화 레이어
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()  # 78. 드롭아웃 레이어 설정 (dropout > 0 일 때 nn.Dropout, 아니면 항등 변환 nn.Identity)
        self.out_conv = nn.Conv2d(c_out, c_out, 3, padding=1)  # 79. 최종 출력 3x3 합성곱 레이어
        nn.init.zeros_(self.out_conv.weight)  # 80. 최종 합성곱 레이어 가중치 0으로 초기화
        nn.init.zeros_(self.out_conv.bias)  # 81. 최종 합성곱 레이어 편향 0으로 초기화
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()  # 82. 잔차 연결(Skip Connection)을 위한 1x1 Conv (입출력 채널이 다를 때 채널 수 맞춤)

    def forward(self, x, cond):  # 83. ResBlock 순전파 메서드
        h = self.in_conv(x)  # 84. 입력 x를 첫 번째 3x3 Conv에 통과
        h = F.silu(self.ada1(h, cond))  # 85. 첫 번째 AdaGN 적용 후 SiLU (Swish) 활성화 함수 통과
        h = self.mid_conv(h)  # 86. 중간 3x3 Conv 통과
        h = F.silu(self.ada2(h, cond))  # 87. 두 번째 AdaGN 적용 후 SiLU 활성화 함수 통과
        h = self.dropout(h)  # 88. 드롭아웃 적용
        h = self.out_conv(h)  # 89. 최종 3x3 Conv 통과
        return F.silu(h + self.skip(x))  # 90. 잔차 연결(h + skip(x))과 SiLU 활성화 함수 적용 후 반환


class Down(nn.Module):  # 91. 다운샘플링 블록 (Downsampling via Strided Convolution)
    def __init__(self, c_in, c_out):  # 92. Down 초기화 (입력 채널, 출력 채널)
        super().__init__()  # 93. 상위 모듈 초기화
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)  # 94. Stride 2인 3x3 Conv로 공간 해상도를 절반(1/2)으로 축소

    def forward(self, x):  # 95. Down 순전파 메서드
        return self.conv(x)  # 96. Conv2d 다운샘플링 실행 후 텐서 반환


class Up(nn.Module):  # 97. 업샘플링 블록 (Upsampling via Nearest Interpolation + Conv2d)
    def __init__(self, c_in, c_out):  # 98. Up 초기화 (입력 채널, 출력 채널)
        super().__init__()  # 99. 상위 모듈 초기화
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # 100. Nearest Neighbor 방식을 사용하여 해상도를 2배로 확대
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)  # 101. 3x3 Conv로 업샘플링 후 채널 수 조정

    def forward(self, x):  # 102. Up 순전파 메서드
        return self.conv(self.up(x))  # 103. 업샘플링 수행 후 3x3 Conv 적용 텐서 반환


class SmallUNet(nn.Module):  # 104. 노이즈 예측을 위한 백본 U-Net 모델 클래스 (SmallUNet)
    """
    Small U-Net Architecture without Attention.
    해상도 변화: 28x28 -> 14x14 -> 7x7 -> 14x14 -> 28x28
    """

    def __init__(self, num_classes=10, null_class=True, ch=64, t_dim=128, y_dim=128, dropout=0.0):  # 105. SmallUNet 초기화 메서드
        super().__init__()  # 106. 상위 모듈 초기화
        self.null_id = num_classes if null_class else None  # 107. null_class가 True이면 null_id를 num_classes(10)로 지정 (Unconditional Token)
        n_embed = num_classes + (1 if null_class else 0)  # 108. 총 임베딩 개수 계산 (클래스 10개 + Null 토큰 1개 = 11개)
        self.class_embed = nn.Embedding(n_embed, y_dim)  # 109. 클래스 레이블 임베딩 테이블 생성 (크기: 11 x y_dim)
        self.t_dim, self.y_dim = t_dim, y_dim  # 110. t_dim과 y_dim 변수 저장
        cond_dim = t_dim + y_dim  # 111. 결합된 조건 차원 정의 (시간 임베딩 차원 + 클래스 임베딩 차원 = 256)

        self.inp = nn.Conv2d(1, ch, 3, padding=1)  # 112. 입력 이미지(1 채널)를 기본 채널(ch=64)로 변환하는 초입 3x3 Conv

        self.rb1 = ResBlock(ch, ch, cond_dim, dropout)  # 113. [Encoder Path - Downsampling] 114. 첫 번째 ResBlock (64 -> 64 채널)
        self.down1 = Down(ch, ch * 2)  # 115. 첫 번째 다운샘플링 (28x28 -> 14x14, 64 -> 128 채널)
        self.rb2 = ResBlock(ch * 2, ch * 2, cond_dim, dropout)  # 116. 두 번째 ResBlock (128 -> 128 채널)
        self.down2 = Down(ch * 2, ch * 4)  # 117. 두 번째 다운샘플링 (14x14 -> 7x7, 128 -> 256 채널)

        self.rb_mid1 = ResBlock(ch * 4, ch * 4, cond_dim, dropout)  # 118. [Bottleneck - Mid] 119. 바틀넥 첫 번째 ResBlock (256 -> 256 채널, 7x7)
        self.rb_mid2 = ResBlock(ch * 4, ch * 4, cond_dim, dropout)  # 120. 바틀넥 두 번째 ResBlock (256 -> 256 채널, 7x7)

        self.up1 = Up(ch * 4, ch * 2)  # 121. [Decoder Path - Upsampling] 122. 첫 번째 업샘플링 (7x7 -> 14x14, 256 -> 128 채널)
        self.rb_up1 = ResBlock(ch * 2 + ch * 2, ch * 2, cond_dim, dropout)  # 123. 업샘플링 특징 맵과 Skip Connection 결합 후 ResBlock (128+128=256 -> 128 채널)
        self.up2 = Up(ch * 2, ch)  # 124. 두 번째 업샘플링 (14x14 -> 28x28, 128 -> 64 채널)
        self.rb_up2 = ResBlock(ch + ch, ch, cond_dim, dropout)  # 125. 업샘플링 특징 맵과 Skip Connection 결합 후 ResBlock (64+64=128 -> 64 채널)

        self.out = nn.Conv2d(ch, 1, 3, padding=1)  # 126. 최종 예측 노이즈 출력을 위한 3x3 Conv (64 채널 -> 1 채널)

        self.proj_t = nn.Sequential(nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))  # 127. 시간 임베딩 프로젝션 MLP (Linear -> SiLU -> Linear)
        self.proj_y = nn.Sequential(nn.Linear(y_dim, y_dim * 4), nn.SiLU(), nn.Linear(y_dim * 4, y_dim))  # 128. 클래스 임베딩 프로젝션 MLP (Linear -> SiLU -> Linear)

    def forward(self, x, t, y):  # 129. SmallUNet 순전파 메서드 (x: 노이즈 이미지, t: 타임스텝, y: 클래스 레이블)
        t_emb = self.proj_t(sinusoidal_time_embedding(t, self.t_dim))  # 130. 타임스텝 t로부터 sinusoidal embedding을 생성한 뒤 proj_t MLP 통과 -> t_emb: (B, t_dim)

        if y is not None:  # 131. 클래스 레이블 y가 제공된 경우 임베딩 테이블에서 룩업
            y_emb = self.class_embed(y)  # 132. y 인덱스에 해당하는 class embedding 추출 -> (B, y_dim)
        else:
            y_emb = torch.zeros(x.size(0), self.y_dim, device=x.device)  # 133. y가 None인 경우 0으로 채워진 텐서 생성

        y_emb = self.proj_y(y_emb)  # 134. 클래스 임베딩을 proj_y MLP에 통과 -> y_emb: (B, y_dim)

        cond = torch.cat([t_emb, y_emb], dim=1)  # 135. 시간 임베딩과 클래스 임베딩을 차원 1 방향으로 연결하여 통합 조건 벡터 생성 -> cond: (B, cond_dim)

        h0 = self.inp(x)  # 136. [인코더 순전파] 137. 입력 이미지 x를 초입 Conv에 통과 -> h0: (B, 64, 28, 28)
        h1 = self.rb1(h0, cond)  # 138. 첫 번째 ResBlock 순전파 -> h1: (B, 64, 28, 28)
        h2 = self.down1(h1)  # 139. 첫 번째 Down샘플링 -> h2: (B, 128, 14, 14)
        h3 = self.rb2(h2, cond)  # 140. 두 번째 ResBlock 순전파 -> h3: (B, 128, 14, 14)
        h4 = self.down2(h3)  # 141. 두 번째 Down샘플링 -> h4: (B, 256, 7, 7)

        h5 = self.rb_mid1(h4, cond)  # 142. [바틀넥 순전파] 143. 바틀넥 첫 번째 ResBlock -> h5: (B, 256, 7, 7)
        h6 = self.rb_mid2(h5, cond)  # 144. 바틀넥 두 번째 ResBlock -> h6: (B, 256, 7, 7)

        u1 = self.up1(h6)  # 145. [디코더 순전파] 146. 첫 번째 Up샘플링 -> u1: (B, 128, 14, 14)
        u1 = torch.cat([u1, h3], dim=1)  # 147. 인코더의 h3 특징 맵과 업샘플링된 u1을 채널 방향(dim=1)으로 스킵 연결 -> u1: (B, 256, 14, 14)
        u1 = self.rb_up1(u1, cond)  # 148. 디코더 첫 번째 ResBlock -> u1: (B, 128, 14, 14)

        u2 = self.up2(u1)  # 149. 두 번째 Up샘플링 -> u2: (B, 64, 28, 28)
        u2 = torch.cat([u2, h1], dim=1)  # 150. 인코더의 h1 특징 맵과 업샘플링된 u2를 채널 방향(dim=1)으로 스킵 연결 -> u2: (B, 128, 28, 28)
        u2 = self.rb_up2(u2, cond)  # 151. 디코더 두 번째 ResBlock -> u2: (B, 64, 28, 28)

        eps = self.out(u2)  # 152. 최종 3x3 Conv를 통과시켜 예측 노이즈 eps 산출 -> eps: (B, 1, 28, 28)
        return eps  # 153. 예측된 노이즈 텐서 반환


# ==============================================================================
# DDPM 손실 함수 (Loss Functions) 구현
# ==============================================================================

def ddpm_loss_epsilon(sched, net, x0, y):  # 154. 표준 DDPM Epsilon 예측 Loss 함수 구현
    """
    표준 ε-예측 DDPM 손실 함수:
      1) t ~ Uniform{0..T-1} 무작위 타임스텝 샘플링
      2) eps ~ N(0, I) 가우시안 노이즈 생성
      3) x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps 생성 (sched.q_sample 사용)
      4) 네트워크로 예측된 eps_hat = net(x_t, t, y) 구함
      5) eps_hat과 실제 eps 간의 MSE 손실 계산
    """
    bsz = x0.size(0)  # 155. 배치 크기 추출
    t = sched.sample_timesteps(bsz)  # 156. 1. Uniform{0..T-1} 범위에서 무작위 타임스텝 t 샘플링 (Shape: (bsz,))

    eps = torch.randn_like(x0)  # 157. 2. x0와 동일한 크기의 가우시안 노이즈 eps ~ N(0, I) 생성

    xt = sched.q_sample(x0, t, eps)  # 158. 3. Forward Process: q_sample을 이용하여 노이즈가 주입된 이미지 x_t 생성

    eps_hat = net(xt, t, y)  # 159. 4. Denoising U-Net 신경망을 호출하여 예측 노이즈 eps_hat 산출 (조건: 실제 레이블 y)

    loss = F.mse_loss(eps_hat, eps)  # 160. 5. 실제 주입된 노이즈 eps와 예측 노이즈 eps_hat 사이의 평균 제곱 오차(MSE) 계산

    return loss  # 161. 계산된 손실(Loss) 스칼라 텐서 반환


def ddpm_loss_cfg(sched, net, x0, y, null_id):  # 162. Classifier-Free Guidance (CFG) 학습 손실 함수 구현
    """
    CFG 훈련 손실 함수 (2-Pass Forward):
      1~3) 동일한 x_t 및 eps 샘플링 과정 진행
      4-1) Conditional Forward Pass: eps_hat_cond = net(x_t, t, y)
      4-2) Unconditional Forward Pass: eps_hat_uncond = net(x_t, t, y_null)
      5) Loss = MSE(eps_hat_cond, eps) + MSE(eps_hat_uncond, eps)
    """
    bsz = x0.size(0)  # 163. 배치 크기 추출
    t = sched.sample_timesteps(bsz)  # 164. 1. 무작위 타임스텝 t 샘플링

    eps = torch.randn_like(x0)  # 165. 2. 가우시안 노이즈 eps 생성

    xt = sched.q_sample(x0, t, eps)  # 166. 3. Forward Process: x_t 샘플링

    eps_hat_cond = net(xt, t, y)  # 167. 4-1. Conditional Forward Pass: 실제 클래스 레이블 y를 사용하여 노이즈 예측

    y_null = torch.full((bsz,), null_id, device=x0.device, dtype=torch.long)  # 168. 4-2. Unconditional Forward Pass를 위한 null_id 텐서 생성 (크기: (bsz,), 값: null_id=10)
    eps_hat_uncond = net(xt, t, y_null)  # 169. Unconditional Forward Pass 실행하여 무조건부 노이즈 예측

    loss = F.mse_loss(eps_hat_cond, eps) + F.mse_loss(eps_hat_uncond, eps)  # 170. 5. Conditional MSE Loss와 Unconditional MSE Loss를 합산

    return loss  # 171. 최종 합산된 손실(Loss) 반환


# ==============================================================================
# EMA (Exponential Moving Average) 최적화 클래스
# ==============================================================================
class EMA:
    def __init__(self, model, decay=0.999):  # 172. EMA 클래스 초기화 (모델 객체, decay 율=0.999)
        self.decay = decay  # 173. 감쇄 비율 저장
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}  # 174. 모델의 초기 파라미터 텐서들을 복사하여 섀도우(shadow) 딕셔너리로 보관

    def update(self, model):  # 175. 매 훈련 스텝마다 섀도우 파라미터를 이동 평균으로 업데이트하는 메서드
        with torch.no_grad():  # 176. 기울기 계산을 수행하지 않음 (torch.no_grad)
            for k, v in model.state_dict().items():  # 177. 현재 모델의 state_dict 아이템 순회
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)  # 178. shadow = decay * shadow + (1 - decay) * current_param 수식 적용

    def copy_to(self, model):  # 179. 평가/샘플링 시 EMA 섀도우 파라미터를 실제 모델로 복사하는 메서드
        model.load_state_dict(self.shadow, strict=True)  # 180. 엄격한 매칭(strict=True) 모드로 shadow 파라미터를 모델에 로드


# ==============================================================================
# 역방향 샘플러 (Samplers) 구현
# ==============================================================================

@torch.no_grad()  # 181. 무조건부 DDPM 역방향 샘플링 함수
def sample_unconditional(net, sched, n=64, steps=None):
    """
    무조건부 역방향 DDPM 샘플링 (Ancestral Sampling):
      - 모든 타임스텝에서 Null 클래스 토큰만 사용
      - t = T-1 부터 t = 0 까지 역순으로 순회
    """
    net.eval()  # 182. 모델을 평가(eval) 모드로 전환
    T = sched.T if steps is None else steps  # 183. steps가 지정되어 있으면 해당값 사용, 아니면 스케줄러의 총 타임스텝 T 사용
    x = torch.randn(n, 1, 28, 28, device=device)  # 184. 완전한 가우시안 표준 노이즈 x ~ N(0, I) 샘플링 (n개, 1채널, 28x28)
    y_null = torch.full((n,), net.null_id, device=device, dtype=torch.long)  # 185. 무조건부 샘플링을 위해 null_id로 가득 찬 레이블 텐서 y_null 생성

    for ti in reversed(range(T)):  # 186. 타임스텝 T-1부터 0까지 역순 루프 수행
        t = torch.full((n,), ti, device=device, dtype=torch.long)  # 187. 현재 타임스텝 ti로 채워진 텐서 t 생성 (크기: (n,))

        eps_hat = net(x, t, y_null)  # 188. 1. U-Net 추론: null_id 조건으로 노이즈 eps_hat 예측

        mu, var = sched.posterior_mean_variance(x, eps_hat, t)  # 189. 2. posterior_mean_variance를 통해 p(x_{t-1} | x_t)의 평균(mu)과 분산(var) 계산

        if ti > 0:  # 190. ti > 0 이면 주입 노이즈 z ~ N(0, I) 추가하여 x_{t-1} 복원
            x = mu + torch.sqrt(var) * torch.randn_like(x)  # 191. x_{t-1} = mu + sqrt(var) * z
        else:
            x = mu  # 192. ti == 0 일 때 (마지막 스텝) 노이즈 추가 없이 평균(mu)만 사용

    return x.clamp(-1, 1)  # 193. 이미지 픽셀값을 [-1.0, 1.0] 범위로 클램핑하여 리턴


@torch.no_grad()  # 194. Classifier-Free Guidance (CFG) 기반 조건부 역방향 샘플링 함수
def sample_conditional(net, sched, label, gamma=3.0, n=64, steps=None):
    """
    Classifier-Free Guidance를 적용한 조건부 DDPM 샘플링:
      - 1스텝당 Conditional(eps_c) 및 Unconditional(eps_u) 2회 Forward 수행
      - CFG 가이던스 공식: eps_hat = (1 + gamma) * eps_c - gamma * eps_u
    """
    net.eval()  # 195. 모델을 평가 모드로 전환
    T = sched.T if steps is None else steps  # 196. 총 타임스텝 수 설정
    x = torch.randn(n, 1, 28, 28, device=device)  # 197. 가우시안 표준 노이즈 x ~ N(0, I) 샘플링 (Shape: (n, 1, 28, 28))
    y_lab = torch.full((n,), int(label), device=device, dtype=torch.long)  # 198. 요청받은 생성 숫자 label(0~9)로 채워진 조건부 레이블 텐서 생성
    y_null = torch.full((n,), net.null_id, device=device, dtype=torch.long)  # 199. null_id(10)로 채워진 무조건부 레이블 텐서 생성

    for ti in reversed(range(T)):  # 200. 타임스텝 T-1부터 0까지 역순 루프 실행
        t = torch.full((n,), ti, device=device, dtype=torch.long)  # 201. 현재 타임스텝 텐서 t 생성

        eps_c = net(x, t, y_lab)  # 202. 1-1. Conditional Forward Pass 추론 (eps_c)
        eps_u = net(x, t, y_null)  # 203. 1-2. Unconditional Forward Pass 추론 (eps_u)

        eps_hat = (1.0 + gamma) * eps_c - gamma * eps_u  # 204. 2. CFG 가이던스 수식 적용: eps_hat = (1.0 + gamma) * eps_c - gamma * eps_u

        mu, var = sched.posterior_mean_variance(x, eps_hat, t)  # 205. 3. 보정된 eps_hat으로 p(x_{t-1} | x_t) 분포의 평균(mu) 및 분산(var) 구함

        if ti > 0:  # 206. ti > 0 일 때 가우시안 노이즈 주입하여 역방향 샘플링 진행
            x = mu + torch.sqrt(var) * torch.randn_like(x)  # 207. x_{t-1} = mu + sqrt(var) * z
        else:
            x = mu  # 208. ti == 0 일 때 노이즈 없이 평균값 리턴

    return x.clamp(-1, 1)  # 209. 최종 생성된 이미지 텐서를 [-1.0, 1.0]으로 클램프 후 반환


# ==============================================================================
# 시각화 유틸리티 함수
# ==============================================================================
def show_grid(x, nrow=8, title=''):
    x = (x.clamp(-1, 1) + 1) * 0.5  # 210. 픽셀값 범위 [-1, 1]을 [0, 1] 범위로 복원
    grid = vutils.make_grid(x, nrow=nrow)  # 211. torchvision make_grid를 사용하여 이미지 그리드 생성
    import matplotlib.pyplot as plt  # 212. matplotlib.pyplot 모듈 임포트
    plt.figure(figsize=(8, 8))  # 213. 8x8 커스텀 플롯 피규어 생성
    plt.axis('off')  # 214. 플롯 축 숨기기
    if title:  # 215. 제목이 주어진 경우 타이틀 설정
        plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).cpu())  # 216. (C, H, W) 형태의 텐서를 (H, W, C) 차원으로 변환하여 이미지 출력
    plt.show()  # 217. 화면에 플롯 시각화


# ==============================================================================
# 메인 실행 데모 파이프라인
# ==============================================================================
if __name__ == '__main__':
    print("=== DDPM & CFG Training Pipeline Start ===")  # 218. 메인 스크립트 실행 알림 출력
    sched = DiffusionSchedule(T=200)  # 219. Diffusion Schedule 객체 생성 (T=200)
    net = SmallUNet(num_classes=num_classes, null_class=True, ch=64).to(device)  # 220. SmallUNet 백본 모델 생성 후 지정된 디바이스로 이동
    ema = EMA(net, decay=0.999)  # 221. EMA 객체 초기화

    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)  # 222. AdamW 옵티마이저 설정 (학습률 lr=1e-4)
    USE_CFG_LOSS = True  # 223. CFG 손실 함수 사용 여부 플래그
    epochs = 3  # 224. 데모용 에포크 수 설정 (3 에포크)

    net.train()  # 225. 모델을 훈련 모드로 전환
    for epoch in range(1, epochs + 1):  # 226. 에포크 수만큼 학습 진행 루프
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):  # 227. DataLoader로부터 배치(x: 이미지, y: 클래스) 순회
            x = x.to(device)  # 228. 이미지 텐서를 디바이스로 이동
            y = y.to(device)  # 229. 레이블 텐서를 디바이스로 이동

            if USE_CFG_LOSS:  # 230. USE_CFG_LOSS 플래그에 따른 손실 함수 계산
                loss = ddpm_loss_cfg(sched, net, x, y, null_id=net.null_id)  # 231. CFG 2-Pass 손실함수 호출
            else:
                loss = ddpm_loss_epsilon(sched, net, x, y)  # 232. 단일 조건부 표준 손실함수 호출

            opt.zero_grad()  # 233. 그래디언트 초기화
            loss.backward()  # 234. 역전파 수행하여 파라미터 그래디언트 계산
            opt.step()  # 235. 옵티마이저 파라미터 업데이트
            ema.update(net)  # 236. EMA 섀도우 파라미터 업데이트

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")  # 237. 에포크 종료 후 현재 손실 출력

    ema.copy_to(net)  # 238. 훈련 완료 후 EMA 파라미터를 모델에 적용 후 평가 모드로 전환
    net.eval()

    print("Generating Unconditional Samples...")  # 239. 무조건부 이미지 샘플링 테스트 (64개 이미지 생성)
    x_uncond = sample_unconditional(net, sched, n=64)
    print(f"Unconditional samples shape: {x_uncond.shape}")

    print("Generating Conditional Samples (Digit=5, Gamma=3.0)...")  # 240. 조건부 CFG 이미지 샘플링 테스트 (숫자 5, gamma=3.0)
    x_cond = sample_conditional(net, sched, label=5, gamma=3.0, n=64)
    print(f"Conditional samples shape: {x_cond.shape}")
