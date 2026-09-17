# magnetron-serve

The CLI and inference server for [magnetron-models](https://github.com/MarioSieg/magnetron-models): install `.mag` snapshots
from the Hugging Face Hub, chat with one in a terminal REPL, or put one behind an SSE HTTP endpoint.

`magnetron-models` holds the modelling code — architectures, tokenizer, KV cache, the streaming
generator and the safetensors conversion pipelines. Everything that faces a user or a socket lives
here.

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
magnetron-serve serve [<model>]      SSE inference server
magnetron-serve rm <model>...        delete a downloaded snapshot
```

A `<model>` is one of three things, and every command takes all three:

* a registry name — `qwen3.5-9b`, one of the curated snapshots
* a Hub repo id — `mario-sieg/Qwen3.5-9B-Magnetron`, any repo holding a `.mag`
* a local path — `./qwen3.5-35b-a3b-bf16.mag`

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

## Serve

```bash
mag serve qwen3.8-27b --host 0.0.0.0 --port 11434
```

The default model loads at startup, unless `--lazy`. The wire format is OpenAI's, so existing
clients work unchanged:

| Endpoint | |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI chat completions, `"stream": true` for SSE |
| `POST /api/generate` | raw prompt in, tokens out, no chat template applied |
| `GET /v1/models` | the registry, with what is installed |
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
default) would be resident.

## How requests are distributed

A model carries its KV cache inside itself and generation saturates the device, so two requests can
neither share an engine nor usefully interleave. The HTTP layer is threaded — many clients stay
connected and stream — while a FIFO gate lets exactly one of them generate at a time, in arrival
order. Each request carries its whole history in the prompt and resets the cache, so no request ever
sees another's turns. Beyond `--queue-limit` waiters, new requests get a `503` rather than an
unbounded wait, and a client that hangs up mid-stream is noticed on the next chunk and its
generation abandoned there.
