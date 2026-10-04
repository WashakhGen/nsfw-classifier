# NSFW Classifier

A FastAPI service that classifies uploaded images as safe or NSFW, using the
[Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector)
model (EVA02, 448px). The model runs on NVIDIA (CUDA), AMD (ROCm), Apple
Silicon (MPS), Intel (XPU) or CPU, and the hardware is detected automatically.

## Features

- Rates each image on four levels: `neutral`, `low`, `medium` and `high`.
- The NSFW level and threshold are configurable.
- The server starts listening immediately while the model loads in the
  background. `/health` reports the loading state.
- `torch.compile` is used where available, with warmup at startup and a
  fallback to the uncompiled model if compilation fails.
- Uploads are checked for file size and image dimensions, which also guards
  against decompression bombs.
- The model downloads on first run to `./models`, pinned to a specific
  revision.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 or newer (installed by uv from `.python-version`)

## Quick start

```bash
uv sync
uv run main.py
```

The server listens on `http://0.0.0.0:8888`. On first run, the model weights
(about 350 MB) are downloaded to `./models/`.

Interactive API docs are at `http://localhost:8888/docs`.

## API

### `GET /health`

Returns the model's loading state.

| Status | Body |
|---|---|
| `200` | `{"status": "ready", "error": null}` |
| `503` | `{"status": "loading", "error": null}` |
| `503` | `{"status": "failed", "error": "<reason>"}` |

Use this endpoint for readiness probes.

### `POST /nsfw-check`

Upload an image as multipart form data in the `file` field.

```bash
curl -F "file=@photo.jpg" http://localhost:8888/nsfw-check
```

Response:

```json
{
  "task_id": "5c3904a7-44d6-4db7-b8c6-387fe6a4827c",
  "is_nsfw": false,
  "classification": {
    "nsfw_score": 0.0003,
    "neutral": 0.9996,
    "low": 0.0004,
    "medium": 0.0003,
    "high": 0.0003
  }
}
```

Errors:

| Status | Cause |
|---|---|
| `400` | The file is not a valid image. |
| `413` | The file is larger than `MAX_FILE_BYTES`, or a side is longer than `MAX_SIDE_PX`. |
| `500` | Inference failed, for example because the device ran out of memory. |
| `503` | The model is still loading, or loading failed. |

## How scoring works

The model outputs a probability for each of the four levels. Following
Freepik's reference implementation, `low` and `medium` in the response are
**cumulative**: each is the probability that the image is at that level *or
worse*.

| Field | Meaning |
|---|---|
| `neutral` | P(neutral) |
| `low` | P(low) + P(medium) + P(high) |
| `medium` | P(medium) + P(high) |
| `high` | P(high) |

Because of this, the fields don't add up to 1.

`nsfw_score` is the cumulative score at the configured `NSFW_LEVEL`. An image
is flagged when `nsfw_score >= NSFW_THRESHOLD`. With the defaults (`medium`,
`0.5`), an image is NSFW if it is more likely than not to be medium or high.

## Configuration

Set any of these in a `.env` file in the project root, or as environment
variables.

| Variable | Default | Description |
|---|---|---|
| `NSFW_LEVEL` | `medium` | Lowest level that counts as NSFW: `low` (strict), `medium` or `high` (lenient). |
| `NSFW_THRESHOLD` | `0.5` | Cumulative probability at which an image is flagged. |
| `MAX_FILE_BYTES` | `10485760` | Largest accepted upload, in bytes (10 MB). |
| `MAX_SIDE_PX` | `4096` | Largest accepted image width or height. |
| `DEVICE` | `auto` | `auto`, `cpu`, `cuda`, `cuda:1`, `mps` or `xpu`. |
| `COMPILE_NSFW_CHECKER` | `true` | Compile the model with `torch.compile`. |
| `MODEL_NAME` | `Freepik/nsfw_image_detector` | Hugging Face model ID. |
| `MODEL_REVISION` | pinned commit | Model commit to download. Use `main` for the latest. |
| `MODELS_PATH` | `./models` | Where model weights are stored. |
| `UVICORN_HOST` | `0.0.0.0` | Address the server binds to. |
| `UVICORN_PORT` | `8888` | Port the server listens on. |

On a server without internet access, set `HF_HUB_OFFLINE=1` to use the
already-downloaded weights without contacting Hugging Face.

## Performance

Measured on Apple Silicon (MPS, bfloat16), one request at a time:

| Mode | Time per request |
|---|---|
| Uncompiled | ~88 ms |
| Compiled | ~57 ms |

Startup takes a few seconds, including compilation and warmup. Later starts
are faster because compiled kernels are cached.

Run a single worker process. Each worker loads its own copy of the model, and
requests within a worker are already processed one at a time on the device.


## License

The model is released by Freepik under the MIT license. See the
[model card](https://huggingface.co/Freepik/nsfw_image_detector) for details.
