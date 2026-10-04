from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import ImageClassificationPipeline, PreTrainedModel
from transformers import pipeline as nsfw_pipeline

from app import hardware
from app.logger import log_main
from app.schema import InferenceError, NSFWResult
from app.settings import SETTINGS


class NSFWCheck:
    def __init__(self) -> None:
        self.safety_checker = self.load_safety_checker()

    def _warmup(self, runs: int = 3) -> None:
        log_main("Warming up safety checker...")
        start = perf_counter()
        image = Image.fromarray(np.random.randint(0, 256, (768, 1024, 3), dtype=np.uint8))
        for _ in range(runs):
            self.safety_checker(image)
        log_main(f"Warmup done in {perf_counter() - start:.1f}s")

    def _compile(self) -> None:
        eager_model = self.safety_checker.model
        mode = "max-autotune-no-cudagraphs" if hardware.vendor() == hardware.HWVendor.CUDA else "default"
        log_main(f"Compiling safety checker (mode={mode})...")
        start = perf_counter()

        try:
            self.safety_checker.model = cast(PreTrainedModel, torch.compile(eager_model, mode=mode))
            self._warmup()
            log_main(f"Compiled in {perf_counter() - start:.1f}s")

        except Exception as e:
            self.safety_checker.model = eager_model
            log_main(f"Compilation failed, using eager model: {e!r}")
            self._warmup()

    @torch.no_grad()
    def load_safety_checker(self, model_name: str = SETTINGS.MODEL_NAME) -> ImageClassificationPipeline:
        log_main(f"Loading safety checker model {model_name}...")
        MODEL_DIR = Path(SETTINGS.MODELS_PATH) / model_name.split("/")[-1]

        snapshot_download(
            repo_id=model_name,
            local_dir=MODEL_DIR,
            allow_patterns=["*.json", "*.safetensors"],
            revision=SETTINGS.MODEL_REVISION,
        )

        used_before, total = hardware.memory_info()
        log_main(
            f"Hardware: {hardware.vendor().name} | device={hardware.device()} | dtype={hardware.dtype()} | "
            f"total={hardware.gib(total)} Available={hardware.gib(total - used_before)}"
        )

        self.safety_checker = nsfw_pipeline(
            task="image-classification",
            model=str(MODEL_DIR),
            device=hardware.device(),
            dtype=hardware.dtype(),
        )

        if SETTINGS.COMPILE_NSFW_CHECKER:
            self._compile()
        else:
            self._warmup()

        used_after, _ = hardware.memory_info()
        log_main(
            f"Model loaded: size={hardware.gib(used_after - used_before)} | "
            f"Available={hardware.gib(total - used_after)}"
        )

        return self.safety_checker

    def run_safety_check(self, image: Image.Image) -> tuple[NSFWResult, float]:
        try:
            start = perf_counter()
            raw = self.safety_checker(image)
            duration = round(perf_counter() - start, 4)
        except torch.OutOfMemoryError as e:
            hardware.empty_cache()  # free cached memory so the next request can succeed
            raise InferenceError("Out of memory during inference") from e
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

        result = cast("list[dict[str, Any]]", raw)  # one image in -> list of {label, score}
        if not result:
            raise InferenceError("Model returned no predictions")

        record = {r["label"]: float(r["score"]) for r in result}

        return NSFWResult(**record), duration
