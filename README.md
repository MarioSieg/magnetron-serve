# magnetron-serve

The CLI and inference server for [magnetron-models](https://github.com/MarioSieg/magnetron-models): install `.mag` snapshots
from the Hugging Face Hub, chat with one in a terminal REPL, paint images with Qwen-Image 2.1, or put
either behind an SSE HTTP endpoint.

`magnetron-models` holds the modelling code — architectures, tokenizer, KV cache, the streaming
generator, the text-to-image pipeline and the safetensors conversion pipelines. Everything that faces
a user or a socket lives here.

## Install

Requires Python >= 3.14. `magnetron-models` is not published yet, so it is pulled straight from GitHub
via `[tool.uv.sources]`; no local checkout needed.

```bash
uv sync
```

Both `magnetron-serve` and the shorter `mag` are installed as entry points.

## Commands

```
magnetron-serve install <model>...   download a snapshot from the Hub
magnetron-serve list [--all]         what is installed, and what could be
magnetron-serve run <model>          interactive chat REPL
magnetron-serve image <model> [...]  paint an image, or open a REPL that paints one per line
magnetron-serve serve [<model>]      SSE inference server
magnetron-serve rm <model>...        delete a downloaded snapshot
```

A `<model>` is one of three things, and every command takes all three:

* a registry name — `qwen3.5-9b` or `qwen-image-2.1`, one of the curated snapshots
* a Hub repo id — `mario-sieg/Qwen3.5-9B-Magnetron`, any repo holding a `.mag`
* a local path — `./qwen3.5-35b-a3b-bf16.mag`

A snapshot is either a chat model or a text-to-image pipeline; `list` shows which, and `run` and
`image` each refuse the other kind with a pointer to the right command. A bare Hub repo id reveals its
kind once the file is on disk.

Downloads land in the ordinary `huggingface_hub` cache, so a snapshot pulled here is the same file
`magnetron-models` would have pulled on its own and nothing is stored twice.

## Run

```bash
mag install qwen3.8-27b
mag run qwen3.8-27b

mag run qwen3.5-35b-a3b --device cpu       # MoE, CPU is the only practical device today
mag run ./qwen3.5-9b-bf16.mag              # a local snapshot runs on its own
mag run qwen3 --prompt 'Explain RoPE briefly.'
```

`run` pulls the snapshot first if it is missing. `--device` defaults to `auto`: the fastest
accelerator this build can see, or the CPU when there is none. Naming a backend is taken at its word
— `--device cuda` on a machine without one is an error rather than a silent hundredfold slowdown —
and picks that backend's fastest device, so `cuda:1` is only worth spelling out to override the
choice. Sampling comes from `--temp`, `--top-k`, `--max-tokens`, `--seed`, `--dtype` and `--system`;
`--help` lists every model name.

Inside the REPL, `/help` lists the commands, `/clear` starts a new conversation and `/exit` leaves.

## Image

```bash
mag image qwen-image-2.1 'A capybara wearing a wizard hat, reading a book by candlelight, oil painting' -o capybara.png
mag image qwen-image-2.1 '...' --width 1344 --height 768 --steps 30 --seed 42 --device cuda
mag image ./qwen-image-2.1-bf16.mag        # no prompt: a REPL that paints one image per line
```

`.png` keeps the alpha channel the model paints, `.jpg` composites over white. `--negative-prompt`
together with `--guidance-scale` above 1 turns on classifier-free guidance; by default it samples
without guidance like the reference. The pipeline is three networks and loads each just in time so
the whole ~31 GiB (bf16) is never resident at once; `--keep-loaded` holds all three when they fit.

Without a prompt the REPL paints one image per line into `<out>-1.png`, `<out>-2.png`, ... with the
seed counting up, and `/seed`, `/size WxH` and `/steps` change the settings between images.

## Serve

```bash
mag serve qwen3.8-27b --host 0.0.0.0 --port 11434
```

The default model loads at startup, unless `--lazy`. The wire format is OpenAI's, so existing
clients work unchanged:

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI chat completions, `"stream": true` for SSE |
| `POST /v1/images/generations` | OpenAI image generation, `"stream": true` for progress over SSE |
| `POST /api/generate` | raw prompt in, tokens out, no chat template applied |
| `POST /api/images` | alias of `/v1/images/generations` |
| `GET /v1/models` | the registry, chat and image models, with what is installed |
| `GET /health` | queue depth and which models are resident |

```bash
curl -N http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "stream": true, "messages": [{"role": "user", "content": "Hi"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='unused')
for chunk in client.chat.completions.create(
    model='qwen3.8-27b',
    messages=[{'role': 'user', 'content': 'Explain RoPE briefly.'}],
    stream=True,
):
    print(chunk.choices[0].delta.content or '', end='', flush=True)
```

A request that names no `model` gets the one `serve` was started with. A request that names a
different one loads it, evicting the least recently used model when more than `--max-loaded` (1 by
default) would be resident. Chat and image models share the pool, so `--max-loaded 2` keeps one of
each around.

### Images

```bash
curl http://127.0.0.1:11434/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen-image-2.1", "prompt": "A capybara in a wizard hat", "size": "1024x1024", "seed": 42}' \
  | jq -r '.data[0].b64_json' | base64 -d > capybara.png
```

```python
image = client.images.generate(model='qwen-image-2.1', prompt='A capybara in a wizard hat', size='1024x1024', response_format='b64_json')
open('capybara.png', 'wb').write(base64.b64decode(image.data[0].b64_json))
```

The response is `{"created", "model", "data": [{"index", "b64_json", "size", "output_format"}]}`. Only
`b64_json` is offered; the server hosts no files. Beyond OpenAI's `prompt`, `n`, `size` and
`output_format` (`png` or `jpeg`), a request may set `width` and `height` in place of `size`,
`steps`, `seed`, `negative_prompt` and `guidance_scale`; whatever it leaves out falls back to the
flags `serve` was started with (`--width`, `--height`, `--steps`, ...). With `n` above 1 the seeds
count up from the requested one.

With `"stream": true` the reply is an SSE stream: one `image_generation.progress` event
(`index`, `step`, `total`) per denoising step, then an `image_generation.completed` event per image
carrying the same fields as a `data` item, then `[DONE]`. A client that hangs up mid-stream stops the
denoising at the next step.

## How requests are distributed

A model carries its KV cache inside itself and generation saturates the device, so two requests can
neither share an engine nor usefully interleave; the same goes for a denoising run. The HTTP layer is threaded — many clients stay
connected and stream — while a FIFO gate lets exactly one of them generate at a time, in arrival
order. Each request carries its whole history in the prompt and resets the cache, so no request ever
sees another's turns. Beyond `--queue-limit` waiters, new requests get a `503` rather than an
unbounded wait, and a client that hangs up mid-stream is noticed on the next chunk and its
generation abandoned there.
