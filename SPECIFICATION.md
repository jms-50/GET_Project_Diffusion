# DDPM & Classifier-Free Guidance (CFG) 시스템 명세서 (Specification)

이 문서는 MNIST 데이터셋 기반의 **DDPM(Denoising Diffusion Probabilistic Model)** 및 **Classifier-Free Guidance (CFG)** 시스템의 수학적 원리, 전체 시스템 아키텍처, 모듈별 구조, 손실 함수 및 샘플링 알고리즘에 대한 상세 기술 명세서입니다.

---

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 MNIST 손글씨 숫자 데이터셋($28 \times 28$, 1채널)에 대해 무조건부(Unconditional) 및 조건부(Conditional) 이미지 생성을 수행하는 확산 모델(Diffusion Model)을 구현한 시스템입니다.

### 주요 특징
* **기반 모델**: DDPM (Denoising Diffusion Probabilistic Models, Ho et al., 2020)
* **조건부 생성 방식**: Classifier-Free Guidance (CFG, Ho & Salimans, 2022)
* **백본 아키텍처**: Adaptive Group Normalization (AdaGN) 기반의 Small U-Net
* **임베딩 방식**: Sinusoidal Time Embedding 및 Class Embedding (Null Token 지원)
* **최적화 기법**: EMA (Exponential Moving Average, decay=0.999), AdamW Optimizer

---

## 2. 수학적 배경 및 알고리즘 명세 (Mathematical Background)

### 2.1 Forward Process (정방향 확산 과정)
원본 데이터 $x_0 \sim q(x_0)$에 각 타임스텝 $t \in \{1, \dots, T\}$마다 가우시안 노이즈를 단계적으로 추가합니다.

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

여기서:
* $\beta_t \in (0, 1)$ : 타임스텝 $t$에서의 노이즈 스케줄 (Linear Schedule: $\beta_1 = 10^{-4} \dots \beta_T = 0.02$)
* $\alpha_t = 1 - \beta_t$
* $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ (누적 곱, Cumulative Product)

Reparameterization Trick을 사용하여 임의의 타임스텝 $t$의 $x_t$를 한번에 직접 샘플링합니다:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

---

### 2.2 Reverse Process & Posterior Distribution (역방향 과정)
노이즈가 추가된 $x_t$로부터 원본 데이터 $x_{t-1}$을 복원하는 역방향 분포 $p_\theta(x_{t-1} | x_t)$는 다음과 같습니다:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

이론적인 Posterior Distribution $q(x_{t-1} | x_t, x_0)$의 평균 $\tilde{\mu}_t$ 및 분산 $\tilde{\beta}_t$는 다음과 같이 정의됩니다:

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

노이즈 예측 신경망 $\epsilon_\theta(x_t, t)$를 사용할 경우, 예측된 평균 $\mu_\theta(x_t, t)$는 다음과 같이 정형화됩니다:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

---

### 2.3 손실 함수 (Loss Functions)

#### (1) Standard DDPM Epsilon Loss ($\mathcal{L}_{\text{simple}}$)
예측된 노이즈 $\epsilon_\theta(x_t, t, y)$와 실제 가우시안 노이즈 $\epsilon$ 간의 평균 제곱 오차(MSE)를 최소화합니다.

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|^2 \right]$$

#### (2) Classifier-Free Guidance (CFG) Training Loss
CFG는 별도의 분류기(Classifier) 없이 조건부 생성과 무조건부 생성을 단일 네트워크로 학습합니다.
* 조건부 Forward Pass: $\epsilon_{\text{cond}} = \epsilon_\theta(x_t, t, y)$ (실제 클래스 레이블 $y$)
* 무조건부 Forward Pass: $\epsilon_{\text{uncond}} = \epsilon_\theta(x_t, t, y_{\text{null}})$ ($y_{\text{null}} = 10$ 클래스 레이블)
* 두 Forward Pass의 MSE 손실 합산:

$$\mathcal{L}_{\text{CFG}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|^2 + \| \epsilon - \epsilon_\theta(x_t, t, y_{\text{null}}) \|^2 \right]$$

---

### 2.4 Classifier-Free Guidance Sampling (CFG 샘플링)
샘플링 도중 가이던스 스케일 파라미터 $\gamma$ (혹은 $w$)를 사용하여 내포된 조건부 확률 경향성을 강하게 부각시킵니다:

$$\tilde{\epsilon}_\theta(x_t, t, y) = (1 + \gamma) \epsilon_\theta(x_t, t, y) - \gamma \epsilon_\theta(x_t, t, y_{\text{null}})$$

* $\gamma = 0$ : 표준 조건부 예측 ($\epsilon_\theta(x_t, t, y)$)
* $\gamma > 0$ : 조건 강조 (클래스 특징 명확화, 다양성 감소)
* $\gamma < 0$ : 조건 반전 또는 조건 억제

---

## 3. 시스템 아키텍처 (System Architecture)

```
+-----------------------------------------------------------------------+
|                           MNIST Dataset                               |
|               Shape: (B, 1, 28, 28), Normalized to [-1, 1]           |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                        DiffusionSchedule (T=200)                      |
|  - Betas: linear(1e-4 -> 0.02)                                        |
|  - Alphas, Alphas_bar, sqrt_alphas_bar, sqrt_one_minus_alphas_bar     |
|  - q_sample(x0, t, eps) -> xt                                         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                          SmallUNet Architecture                       |
|                                                                       |
|   Inputs: xt (B,1,28,28), t (B,), y (B,)                              |
|                                                                       |
|   [Condition Embedding]                                               |
|   t -> sinusoidal_time_embedding -> proj_t \                          |
|                                              +--> cond (B, 256)       |
|   y -> class_embed (11 classes) -> proj_y  /                          |
|                                                                       |
|   [U-Net Pipeline]                                                    |
|   Input (1ch) -> Conv2d -> h0 (64ch, 28x28)                           |
|      |                                                                |
|      v                                                                |
|   ResBlock1(64->64) -> Down1(64->128) -> ResBlock2(128->128)          |
|      |                                   |                            |
|      v                                   v                            |
|   Down2(128->256) -------------> ResBlock_Mid1, Mid2 (256, 7x7)       |
|                                          |                            |
|                                          v                            |
|   ResBlock_Up1(256+256->128) <--- Up1(256->128)                       |
|      |                                                                |
|      v                                                                |
|   ResBlock_Up2(128+64->64)   <--- Up2(128->64)                        |
|      |                                                                |
|      v                                                                |
|   Out Conv2d (64->1) ----------> eps_pred (B,1,28,28)                  |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                         Sampling / Optimization                       |
|  - Loss: MSE(eps_hat, eps)                                            |
|  - Optimizer: AdamW (lr=1e-4) + EMA (decay=0.999)                     |
|  - Ancestral Reverse Sampling with CFG Adjustment                     |
+-----------------------------------------------------------------------+
```

---

## 4. 컴포넌트 상세 명세 (Component Details)

### 4.1 `DiffusionSchedule`
* **역할**: 전방 확산 모의 및 역방향 Posterior 산출을 위한 상수 계산 및 등록.
* **주요 메서드**:
  * `sample_timesteps(bsz)`: $[0, T-1]$ 범위에서 대칭 균일 분포로 $t$ 생성.
  * `q_sample(x0, t, eps)`: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 계산.
  * `posterior_mean_variance(xt, eps_pred, t)`: 예측 노이즈 $\epsilon_{\text{pred}}$로부터 평균 $\mu$ 및 분산 $\tilde{\beta}_t$ 복원.

---

### 4.2 `sinusoidal_time_embedding(t, dim)`
* **역할**: Transformer 스타일 위치 임베딩 기법을 시간 스텝 $t$에 적용.
* **수식**:
  $$\text{freqs}_i = \exp\left(-\frac{\ln(10000) \cdot i}{\text{dim}/2}\right)$$
  $$\text{emb} = [\sin(t \cdot \text{freqs}), \cos(t \cdot \text{freqs})]$$

---

### 4.3 `AdaGN (Adaptive GroupNorm)`
* **역할**: 시간 및 클래스 조건 벡터 `cond`를 이용하여 채널별 변량(Scale $s$, Shift $b$)을 동적으로 인가.
* **연산**:
  $$\text{AdaGN}(x, \text{cond}) = \text{GroupNorm}(x) \cdot (1 + s) + b$$
  여기서 $[s, b] = \text{Linear}(\text{cond})$.

---

### 4.4 `ResBlock`
* **역할**: Convolution, AdaGN, SiLU 활성화 함수, Dropout 및 Residual Skip Connection을 통합한 잔차 블록.
* **흐름**:
  $$\text{Conv3x3}(x) \to \text{AdaGN} \to \text{SiLU} \to \text{Conv3x3} \to \text{AdaGN} \to \text{SiLU} \to \text{Dropout} \to \text{Conv3x3} + \text{Skip}(x) \to \text{SiLU}$$

---

### 4.5 `SmallUNet`
* **채널 구성**: Input (1) $\to$ Base (64) $\to$ Level 1 (128) $\to$ Level 2 (256)
* **해상도 변화**: $28 \times 28 \to 14 \times 14 \to 7 \times 7 \to 14 \times 14 \to 28 \times 28$
* **클래스 임베딩**:
  * 클래스 개수: 10개 (숫자 0~9)
  * Null 클래스 ID: 10 (`null_id = 10`)
  * 총 임베딩 개수: 11개

---

### 4.6 `EMA (Exponential Moving Average)`
* **역할**: 학습 도중 모델 파라미터의 그림자 복사본(Shadow Parameters)을 관리하여 가중치 진동을 완화하고 생성이미지의 품질 향상.
* **업데이트 수식**:
  $$\theta_{\text{shadow}} \leftarrow \text{decay} \cdot \theta_{\text{shadow}} + (1 - \text{decay}) \cdot \theta_{\text{current}}$$

---

## 5. 핵심 구현 함수 명세 (Core Functions Specification)

### 5.1 `ddpm_loss_epsilon(sched, net, x0, y)`
```python
def ddpm_loss_epsilon(sched, net, x0, y):
    bsz = x0.size(0)
    t = sched.sample_timesteps(bsz)
    eps = torch.randn_like(x0)
    xt = sched.q_sample(x0, t, eps)
    eps_hat = net(xt, t, y)
    loss = F.mse_loss(eps_hat, eps)
    return loss
```
1. 배치 크기 `bsz`만큼 임의의 타임스텝 $t \sim \text{Uniform}(0, T-1)$ 샘플링.
2. 표준 정규분포 노이즈 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 생성.
3. `q_sample`을 이용해 노이즈가 섞인 $x_t$ 산출.
4. U-Net 추론으로 예측 노이즈 $\hat{\epsilon}$ 산출.
5. $\hat{\epsilon}$과 실제 노이즈 $\epsilon$의 MSE 손실 리턴.

---

### 5.2 `ddpm_loss_cfg(sched, net, x0, y, null_id)`
```python
def ddpm_loss_cfg(sched, net, x0, y, null_id):
    bsz = x0.size(0)
    t = sched.sample_timesteps(bsz)
    eps = torch.randn_like(x0)
    xt = sched.q_sample(x0, t, eps)
    eps_hat_cond = net(xt, t, y)
    y_null = torch.full((bsz,), null_id, device=x0.device, dtype=torch.long)
    eps_hat_uncond = net(xt, t, y_null)
    loss = F.mse_loss(eps_hat_cond, eps) + F.mse_loss(eps_hat_uncond, eps)
    return loss
```
1. 동일한 $x_t$와 $\epsilon$에 대해 2회 Forward Pass 수행.
2. `eps_hat_cond`: 실제 레이블 $y$ 조건 적용.
3. `eps_hat_uncond`: Null 레이블 $y_{\text{null}} = 10$ 조건 적용.
4. 조건부 손실과 무조건부 손실을 합산하여 반환.

---

### 5.3 `sample_unconditional(n=64, steps=None)`
```python
@torch.no_grad()
def sample_unconditional(n=64, steps=None):
    net.eval()
    T = sched.T if steps is None else steps
    x = torch.randn(n, 1, 28, 28, device=device)
    y_null = torch.full((n,), net.null_id, device=device, dtype=torch.long)

    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        eps_hat = net(x, t, y_null)
        mu, var = sched.posterior_mean_variance(x, eps_hat, t)
        if ti > 0:
            x = mu + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mu
    return x.clamp(-1,1)
```
* 무조건부(Null Class) 조건만 사용하여 $t=T-1$부터 $t=0$까지 역방향으로 노이즈를 제거하며 샘플링.

---

### 5.4 `sample_conditional(label, gamma=3.0, n=64, steps=None)`
```python
@torch.no_grad()
def sample_conditional(label, gamma=3.0, n=64, steps=None):
    net.eval()
    T = sched.T if steps is None else steps
    x = torch.randn(n, 1, 28, 28, device=device)
    y_lab  = torch.full((n,), int(label), device=device, dtype=torch.long)
    y_null = torch.full((n,), net.null_id, device=device, dtype=torch.long)

    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        eps_c = net(x, t, y_lab)
        eps_u = net(x, t, y_null)
        eps_hat = (1.0 + gamma) * eps_c - gamma * eps_u
        mu, var = sched.posterior_mean_variance(x, eps_hat, t)
        if ti > 0:
            x = mu + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mu
    return x.clamp(-1,1)
```
* CFG 수식 $\hat{\epsilon} = (1+\gamma)\epsilon_c - \gamma \epsilon_u$를 적용하여 샘플링 수행.

---

## 6. 하이퍼파라미터 및 CFG 스케일($\gamma$) 분석 가이드

| $\gamma$ 값 | 특성 및 이미지 변화 경향 |
| :--- | :--- |
| **$\gamma = -1.0$** | 조건 반전(Negative Guidance): 해당 숫자의 반대 특성 또는 배경 위주 생성 |
| **$\gamma = 0.0$** | 표준 조건부 생성: 가이던스 부스팅 없음 (기본 조건 성능) |
| **$\gamma = 1.0 \sim 2.0$** | 균형 잡힌 가이던스: 선명도 향상 및 적절한 다양성 유효 |
| **$\gamma = 3.0 \sim 4.0$** | 추천(Optimal Range): MNIST 선명도가 매우 우수하며 숫자 획이 또렷함 |
| **$\gamma \ge 5.0$** | 과도한 가이던스 (Over-saturation): 콘트라스트가 지나치게 커지고 외곽선 왜곡 발생 가능 |

---

## 7. 결론 및 참고 문헌

1. **Ho, J., Jain, A., & Abbeel, P. (2020).** *Denoising Diffusion Probabilistic Models.* Advances in Neural Information Processing Systems (NeurIPS).
2. **Ho, J., & Salimans, T. (2022).** *Classifier-Free Diffusion Guidance.* arXiv preprint arXiv:2208.11970.
