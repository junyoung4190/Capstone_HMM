"""
FaceShield 보호 처리 통합 진입점
백엔드에서 이거 하나만 import해서 쓰면 됨.

사용 예:
    from pipeline import protect_image

    # 1. 파일 경로
    result = protect_image("/path/to/image.jpg")

    # 2. 이미지 bytes (백엔드 업로드 처리)
    result = protect_image(image_bytes)

    # 3. numpy 배열 (RGB)
    result = protect_image(numpy_array)

    # 4. PIL 이미지
    result = protect_image(pil_image)

    print(result)
    # {
    #   "success": True,
    #   "protected_bytes": b"...",   # 보호된 이미지 PNG bytes
    #   "metrics": {"px_diff": 2.5, ...},
    #   "error": None
    # }
"""

import os
import io
import datetime
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

# Transformers / Diffusers
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL

# IP-Adapter
from utils.unet.ip_adapter.ip_adapter import ImageProjModel
from utils.unet.ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from utils.unet.ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
        AttnProcessor2_0 as AttnProcessor,
    )
else:
    from utils.unet.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# Utility functions
from utils.utils import (
    get_loss_function,
    compute_vae_encodings,
    scale_tensor,
    create_line_mask,
    apply_gaussian,
    AttentionStore,
    compute_contrast_weight,
    generate_face_mask,
    resize_face3,
)

# Attack modules
from utils.unet.unet_attack import AttackUnet_IP_all, AttackCLIP
from utils.landmark.mtcnn_attack import mtcnn_attack
from utils.landmark.arcface_attack import AttackArcFace

# DCT tools
from utils.dct import dct_pass_filter, make_dct_basis, blockfy, encode, decode, deblockfy


# ============================================================
# 1. 설정 (환경 변수로 외부에서 변경 가능)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 경로 (환경 변수로 오버라이드 가능)
MODEL_PATH = os.environ.get("MODEL_PATH", "runwayml/stable-diffusion-v1-5")
UNET_CONFIG = os.environ.get("UNET_CONFIG", "./configs/unet_config.yaml")
IP_ADAPTER_PATH = os.environ.get("IP_ADAPTER_PATH", "./models/ip-adapter_sd15.bin")
IMAGE_ENCODER_PATH = os.environ.get("IMAGE_ENCODER_PATH", "h94/IP-Adapter")
ARCFACE50_PATH = os.environ.get("ARCFACE50_PATH", "./models/arcface_50.pth")
ARCFACE100_PATH = os.environ.get("ARCFACE100_PATH", "./models/arcface_100.pth")
LANDMARK_PATH = os.environ.get(
    "LANDMARK_PATH", "./shape_predictor_68_face_landmarks.dat"
)

# PGD 하이퍼파라미터
PGD_CONFIG = {
    "total_iter": 30,
    "noise_clamp": 12,
    "step_size": 1.0,
    "resize_shape": 512,
    "attn_threshold": 0.2,
    "proj_func": "l1",
    "attn_func": "l2",
    "mtcnn_func": False,  # 원본 default
    "arc_func": "cosine",
    "lambda_face": 1.5,  # v4 multibranch
    "lambda_bg": 1.0,
    "dilation": 15,
    "blur_kernel": 15,
}


# ============================================================
# 2. 모델 로딩 (모듈 import 시 1번만)
# ============================================================
print("[FaceShield] Loading models...")

# 2-1. Config
_config = OmegaConf.load(UNET_CONFIG)

# 2-2. Stable Diffusion 관련
_tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
_text_encoder = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder")
_vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae")
_unet = AttackUnet_IP_all.from_pretrained(
    MODEL_PATH, subfolder="unet", config_file=_config, strict=False
)

# 2-3. CLIP Image Encoder
_image_preprocess = AttackCLIP()
_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    IMAGE_ENCODER_PATH, subfolder="models/image_encoder"
)

# 2-4. ArcFace (얼굴 인식)
_face_embedder50 = torch.load(ARCFACE50_PATH, weights_only=False)
_face_embedder100 = torch.load(ARCFACE100_PATH, weights_only=False)
_id_preprocess = AttackArcFace()

# 2-5. requires_grad False + GPU 이동
_vae.requires_grad_(False).to(device)
_text_encoder.requires_grad_(False).to(device)
_image_encoder.requires_grad_(False).to(device)
_face_embedder50.requires_grad_(False).to(device)
_face_embedder100.requires_grad_(False).to(device)

# 2-6. IP-Adapter Image Projection Model
_image_proj_model = ImageProjModel(
    cross_attention_dim=_unet.config.cross_attention_dim,
    clip_embeddings_dim=_image_encoder.config.projection_dim,
    clip_extra_context_tokens=4,
).to(device)

# 2-7. UNet Attention Processors
_attn_procs = {}
_unet_sd = _unet.state_dict()
for name in _unet.attn_processors.keys():
    cross_attention_dim = (
        None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
    )
    if name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = _unet.config.block_out_channels[block_id]
    elif name.startswith("mid_block"):
        hidden_size = _unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]

    if cross_attention_dim is None:
        _attn_procs[name] = AttnProcessor()
    else:
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_ip.weight": _unet_sd[layer_name + ".to_k.weight"],
            "to_v_ip.weight": _unet_sd[layer_name + ".to_v.weight"],
        }
        _attn_procs[name] = IPAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
        _attn_procs[name].load_state_dict(weights)

_unet.set_attn_processor(_attn_procs)
_adapter_modules = torch.nn.ModuleList(_unet.attn_processors.values())

# 2-8. IP-Adapter 가중치 로드
_state_dict = torch.load(IP_ADAPTER_PATH, map_location=device, weights_only=True)
_image_proj_model.load_state_dict(_state_dict["image_proj"], strict=True)
_adapter_modules.load_state_dict(_state_dict["ip_adapter"], strict=True)

_unet.requires_grad_(False).to(device)

# 2-9. Text Encoder Hidden States (empty prompt)
_inputs = _tokenizer(
    [""],
    max_length=_tokenizer.model_max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
_encoder_hidden_states = _text_encoder(_inputs.input_ids.to(device))[0]

print("[FaceShield] Models loaded successfully.")


# ============================================================
# 3. 헬퍼 함수
# ============================================================
def _to_pil(image_input):
    """다양한 입력 형식 → PIL.Image (RGB)"""
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if hasattr(image_input, "read"):  # file-like
        return Image.open(image_input).convert("RGB")
    raise ValueError(f"Unsupported input type: {type(image_input)}")


def _pil_to_tensor(pil_img, size=512):
    """PIL → (1, 3, H, W) tensor [0, 1] range"""
    pil_img = pil_img.resize((size, size), Image.LANCZOS)
    arr = np.array(pil_img).astype(np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(device)


def _tensor_to_bytes(tensor, format="PNG"):
    """(1, 3, H, W) tensor [0, 1] → PNG bytes"""
    arr = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()


def _compute_metrics(gt_face, protected):
    """간단한 메트릭 계산 (pixel_diff)"""
    diff = (gt_face - protected).abs() * 255
    return {"px_diff": float(diff.mean().cpu())}


# ============================================================
# 4. PGD 보호 처리 핵심 로직 (attack.py에서 분리)
# ============================================================
def _run_pgd(gt_face):
    """
    v4 Multi-branch PGD 보호 처리.

    Args:
        gt_face: (1, 3, H, W) tensor, [0, 1] range

    Returns:
        protected: (1, 3, H, W) tensor, [0, 1] range
    """
    # === 얼굴 마스크 생성 (Multi-branch용) ===
    face_mask = generate_face_mask(
        gt_face[0],
        LANDMARK_PATH,
        dilation=PGD_CONFIG["dilation"],
        blur_kernel=PGD_CONFIG["blur_kernel"],
    )

    # === Loss 함수 ===
    proj_func = get_loss_function(PGD_CONFIG["proj_func"])
    attn_func = get_loss_function(PGD_CONFIG["attn_func"])
    mtcnn_func = get_loss_function(PGD_CONFIG["mtcnn_func"])
    arc_func = get_loss_function(PGD_CONFIG["arc_func"])

    # === VAE Encoding ===
    latents_query = compute_vae_encodings(gt_face, _vae, device, gt=True)

    # === DCT 준비 ===
    N = 8
    DCT_basis = make_dct_basis(N, device)
    low_pass_filter, _ = dct_pass_filter(device)
    timestep = torch.tensor([0], device=device)

    # === ArcFace GT ===
    gt_50, gt_100 = _id_preprocess.preprocess(gt_face)
    gt_id_50 = _face_embedder50(gt_50.to(device))
    gt_id_100 = _face_embedder100(gt_100.to(device))

    # === UNet GT ===
    var_controller = AttentionStore()
    gt_preprocessed = _image_preprocess(gt_face)
    gt_encoded = _image_encoder(gt_preprocessed).image_embeds
    gt_proj = _image_proj_model(gt_encoded)
    stacked_encoder_hidden_states = torch.cat([_encoder_hidden_states, gt_proj], dim=1)
    _unet(
        latents_query,
        timestep,
        stacked_encoder_hidden_states,
        store_controller=var_controller,
        unet_threshold=PGD_CONFIG["attn_threshold"],
    )

    # === PGD 30 iter ===
    with torch.enable_grad():
        delta = torch.zeros_like(gt_face, requires_grad=True).to(device)

        # Contrast scaling 활성화
        with torch.no_grad():
            contrast_weight = compute_contrast_weight(gt_face)

        for i in tqdm(range(PGD_CONFIG["total_iter"]), desc="[PGD]"):
            adv_face = (255 * gt_face) + delta
            adv_face = torch.clamp(adv_face, min=0, max=255)

            # MTCNN attack
            mtcnn_loss = 0
            mtcnn_loss = mtcnn_attack(
                2 * (adv_face / 255) - 1, loss_fn=mtcnn_func, loss=mtcnn_loss, device=device
            )

            # ArcFace Identity Attack
            adv_50, adv_100 = _id_preprocess.preprocess(adv_face / 255)
            adv_id_50 = _face_embedder50(adv_50)
            adv_id_100 = _face_embedder100(adv_100)
            id_loss_50 = arc_func(adv_id_50, gt_id_50)
            id_loss_100 = arc_func(adv_id_100, gt_id_100)
            id_loss = (-1) * id_loss_50 + (-1) * id_loss_100

            # Diff-Conditioned UNet Attack
            adv_preprocessed = _image_preprocess(adv_face / 255)
            adv_encoded = _image_encoder(adv_preprocessed).image_embeds
            adv_proj = _image_proj_model(adv_encoded)
            stacked_encoder_hidden_states = torch.cat(
                [_encoder_hidden_states, adv_proj], dim=1
            )

            clip_loss = proj_func(adv_encoded, gt_encoded)
            attn_loss = 0
            attn_loss = _unet(
                latents_query,
                timestep,
                stacked_encoder_hidden_states,
                loss_fn=attn_func,
                loss=attn_loss,
                gt_attn_map=var_controller.attn_map.copy(),
            )
            unet_loss = (-1) * clip_loss + (+1) * attn_loss

            # PGD Update with Multi-branch region weight
            total_loss = 9 * mtcnn_loss + 4 * id_loss + 1 * unet_loss
            total_loss.backward(retain_graph=True)

            # ★ Multi-branch: 영역별 gradient 가중치 ★
            region_weight = (
                PGD_CONFIG["lambda_face"] * face_mask
                + PGD_CONFIG["lambda_bg"] * (1 - face_mask)
            )
            new_delta = (
                PGD_CONFIG["step_size"] * torch.sign(delta.grad) * region_weight
            )

            # Smooth with Gaussian Blur
            d_rgb = scale_tensor(new_delta)
            mask = create_line_mask(None, d_rgb)
            new_delta = apply_gaussian(None, new_delta, mask, 9, 5)

            # Low-Pass Filter in DCT Domain
            delta.data -= new_delta
            grad_block, pad_size = blockfy(delta.data, N)
            grad_dct = encode(grad_block, DCT_basis)
            grad_dct_passed = grad_dct * low_pass_filter.expand(grad_dct.shape)
            grad_block_passed = decode(grad_dct_passed, DCT_basis)
            delta.data = deblockfy(grad_block_passed, pad_size)

            # Contrast-aware per-pixel clamp
            if contrast_weight is not None:
                local_max = PGD_CONFIG["noise_clamp"] * contrast_weight
                delta.data = torch.clamp(delta.data, min=-local_max, max=local_max)
            else:
                delta.data = torch.clamp(
                    delta.data,
                    min=-PGD_CONFIG["noise_clamp"],
                    max=PGD_CONFIG["noise_clamp"],
                )

            # Final clamp
            delta.data = torch.clamp(
                delta.data,
                min=-PGD_CONFIG["noise_clamp"],
                max=PGD_CONFIG["noise_clamp"],
            )
            delta.grad = None

            # 메모리 정리
            del mtcnn_loss, clip_loss, attn_loss, unet_loss, total_loss
            del id_loss_50, id_loss_100, id_loss
            torch.cuda.empty_cache()

    # === 최종 보호 이미지 ===
    protected = torch.clamp((gt_face * 255) + delta, 0, 255) / 255
    return protected


# ============================================================
# 5. 메인 API (백엔드가 호출하는 함수)
# ============================================================
def protect_image(image_input):
    """
    이미지에 적대적 노이즈를 추가하여 deepfake 보호 처리.

    Args:
        image_input: 파일경로(str) / bytes / numpy.ndarray / PIL.Image / file-like

    Returns:
        dict: {
            "success" (bool),
            "protected_bytes" (bytes or None): PNG 포맷 보호된 이미지
            "metrics" (dict): {"px_diff": float}
            "error" (str or None)
        }
    """
    result = {
        "success": False,
        "protected_bytes": None,
        "metrics": None,
        "error": None,
    }

    try:
        # 1. 입력 → PIL → tensor
        pil_img = _to_pil(image_input)
        gt_face = _pil_to_tensor(pil_img, size=PGD_CONFIG["resize_shape"])

        # 2. PGD 보호 처리
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            protected = _run_pgd(gt_face)

        # 3. 결과 metric 계산
        metrics = _compute_metrics(gt_face, protected)

        # 4. tensor → PNG bytes
        protected_bytes = _tensor_to_bytes(protected, format="PNG")

        result.update(
            {
                "success": True,
                "protected_bytes": protected_bytes,
                "metrics": metrics,
            }
        )

    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    return result


# ============================================================
# 6. 테스트 (모듈 직접 실행 시)
# ============================================================
if __name__ == "__main__":
    test_path = os.environ.get("TEST_IMAGE", "./data/test/Tom_ori.jpg")
    print(f"[테스트 입력] {test_path}\n")

    result = protect_image(test_path)

    if result["success"]:
        # 결과 이미지 저장
        out_path = "./pipeline_test_output.png"
        with open(out_path, "wb") as f:
            f.write(result["protected_bytes"])
        print(f"[OK] saved to {out_path}")
        print(f"[Metrics] {result['metrics']}")
        print(f"[Size] {len(result['protected_bytes']) / 1024:.1f} KB")
    else:
        print(f"[ERROR]\n{result['error']}")