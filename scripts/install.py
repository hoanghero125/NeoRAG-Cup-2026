"""Cài môi trường cho NeoRAG Cup 2026.

Chạy trong môi trường ảo (.venv hoặc conda) đã kích hoạt, tại thư mục gốc của repo:

    python scripts/install.py           # baseline: embedding qua gateway, không cần PyTorch
    python scripts/install.py --local   # thêm PyTorch + sentence-transformers để chạy model trên máy

Với --local: `pip install` thông thường trên Windows luôn cài PyTorch bản CPU, kể cả khi máy có
GPU. Script kiểm tra `nvidia-smi`, cài PyTorch bản CUDA phù hợp với driver, rồi mới cài repo.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

# Các bản CUDA PyTorch phát hành wheel, từ mới đến cũ
CUDA_TAGS = ["cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118"]
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"


def run(*args: str) -> int:
    print("$", " ".join(args), flush=True)
    return subprocess.call(args)


def pip(*args: str) -> int:
    return run(sys.executable, "-m", "pip", *args)


def driver_cuda_version() -> tuple[int, int] | None:
    """Phiên bản CUDA tối đa driver NVIDIA hỗ trợ, hoặc None nếu không có GPU NVIDIA."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else None


def torch_status() -> str:
    """'cuda', 'cpu' hoặc 'missing' cho PyTorch đang cài trong môi trường này."""
    code = "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "missing"


def compatible_tags(driver: tuple[int, int]) -> list[str]:
    """Các bản CUDA không mới hơn driver, ví dụ (12, 8) -> cu128, cu126, ..."""
    return [t for t in CUDA_TAGS if (int(t[2:-1]), int(t[-1])) <= driver]


def install_cuda_torch(driver: tuple[int, int]) -> bool:
    for tag in compatible_tags(driver):
        print(f"\n→ Thử PyTorch bản {tag}")
        if pip("install", "torch", "--index-url", TORCH_INDEX.format(tag=tag)) == 0:
            return True
    return False


def main() -> int:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env == "base":
        print(
            "Cảnh báo: đang ở môi trường conda base. Nên tạo môi trường riêng (xem docs/guide.md)."
        )
    elif sys.prefix == sys.base_prefix and not conda_env:
        print(
            "Cảnh báo: chưa kích hoạt môi trường ảo (.venv hoặc conda). Nên tạo và kích hoạt trước khi cài."
        )

    if "--local" not in sys.argv[1:]:
        if pip("install", "-e", ".[app,dev]") != 0:
            return 1
        print("Cài đặt xong. Thử: neorag --help")
        return 0

    driver = driver_cuda_version()
    if driver is None:
        print("Không thấy GPU NVIDIA (nvidia-smi). Cài bản CPU.")
    else:
        print(f"Thấy GPU NVIDIA, driver hỗ trợ CUDA {driver[0]}.{driver[1]}.")
        if torch_status() == "cuda":
            print("PyTorch bản GPU đã có sẵn, giữ nguyên.")
        else:
            if torch_status() == "cpu":
                pip("uninstall", "-y", "torch")
            if not install_cuda_torch(driver):
                print("Không cài được PyTorch bản CUDA phù hợp. Hãy cập nhật driver NVIDIA.")
                return 1

    if pip("install", "-e", ".[app,dev,local]") != 0:
        return 1

    status = torch_status()
    print(f"\nPyTorch: {status.upper()}")
    if driver is not None and status != "cuda":
        print("Lỗi: máy có GPU NVIDIA nhưng PyTorch không dùng được CUDA.")
        return 1
    print("Cài đặt xong. Thử: neorag --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
