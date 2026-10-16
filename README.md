# anine

A local AI assistant for your terminal. A model running on your own machine (via [Ollama](https://ollama.com))
can read files, write and run code, and help with everyday tasks. Real output is fed back to it, so it stops guessing.
Nothing leaves your computer.

## Features

- **Acts on your machine:** reads, searches, edits and writes files, runs Python and shell, runs your project's tests.
- **Sandboxed by default:** code runs in [bubblewrap](https://github.com/containers/bubblewrap) with no network and a read-only home.
- **Asks first:** you approve code runs and file writes; `/undo` reverts the last file change.
- **Works with most Ollama models:** models without reliable native tool calling are detected and driven with plain text action blocks instead.
- **Full-screen TUI** with themes, session save/load, markdown export and `@file` attachments, or a plain REPL with `--no-tui`.
- **Fast on modest hardware:** the model is preloaded at startup, old turns are trimmed in batches to keep Ollama's prompt cache valid, and text-mode generation stops as soon as an action is complete.

## Install

1. Install [Ollama](https://ollama.com) and pull a model, e.g. `ollama pull qwen2.5-coder:7b`.
2. Install anine (Python 3.9+, no dependencies; developed and tested on Linux, macOS should work without the sandbox but is untested):
   ```
   pipx install .          # or: pip install .   (or just copy anine.py somewhere on your PATH)
   ```
3. Optional but recommended on Linux: install `bubblewrap` (`sudo apt install bubblewrap` / `sudo pacman -S bubblewrap`) so code runs in a sandbox.
4. Check your setup: `anine --doctor`

## Use

```
anine                              full-screen chat (type /help)
anine "explain @app.py"            one-shot; @file attaches a file
cat error.log | anine "what went wrong?"
anine -m llama3.1:8b               pick a model;  anine --set-default MODEL  remembers it
anine -c                           continue the last session
```

Handy commands inside a session:

| Command | What it does |
| --- | --- |
| `/help` | list every command |
| `/model [name]`, `/models` | show, switch or pick a model |
| `/review @dir`, `/explain @file`, `/tests @file` | ready-made tasks |
| `/undo` | revert the last file anine wrote |
| `/compact` | summarize old turns to free context |
| `/rules <text>` | standing instructions (persist with `/config save`) |

## Speed

Local models are slow, so a few settings are worth knowing:

| Command | Use it to |
| --- | --- |
| `/bench` | time model load, prompt speed and generation speed, and see whether the model runs on GPU or CPU |
| `/ctx N` | set the context size; smaller (e.g. `4096`) is faster and uses less memory |
| `/think off` | skip hidden reasoning tokens on thinking models (`auto` leaves the model default) |
| `/unload` | free the model's memory now |
| `--tools text\|native\|auto` | choose how the model acts (`auto` is the default) |

`anine --doctor` also shows what Ollama has loaded and where it runs. For a larger context in less memory, start the
Ollama server with `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`.

## Safety

- Code and shell commands run inside a bubblewrap sandbox: no network by default, your home directory read-only
  (`--home off|ro|rw`), and keys, `.env` files, browser data and shell histories hidden.
- anine asks before running code or writing files (`-y` skips this; sudo/rm -rf/force-push style commands
  are always confirmed when unsandboxed or when home is writable).
- Every action is logged to `~/.anine/audit.log`; `/undo` reverts the last file change; backups are in `~/.anine/backups`.
- The model is told to refuse malware and attacks on systems you don't own, and to treat file contents as data, not instructions.
- Without `bwrap` (or with `--no-sandbox`) code runs with your full user privileges. Local models make mistakes: read what you approve.

## Configuration

Settings live in `~/.anine/config.json` (`anine --show-config`). Use `/config save` to persist the current session's settings.
Put project-specific instructions in an `ANINE.md` file in your project folder and anine reads it automatically.

## License

MIT
