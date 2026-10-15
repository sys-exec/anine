# anine

A local AI assistant for your terminal. A model running on your own machine (via [Ollama](https://ollama.com))
can read files, write and run code, and help with everyday tasks, with real output fed back to it so it stops guessing.
Nothing leaves your computer.

## Install

1. Install [Ollama](https://ollama.com) and pull a model, e.g. `ollama pull qwen2.5-coder:7b`
2. Install anine (Python 3.9+, no dependencies; developed and tested on Linux; macOS should work without the sandbox but is untested):
   ```
   pipx install .          # or: pip install .   (or just copy anine.py somewhere on your PATH)
   ```
3. Optional but recommended on Linux: install `bubblewrap` (`sudo apt install bubblewrap` / `sudo pacman -S bubblewrap`)
   so code runs in a sandbox.
4. Check your setup: `anine --doctor`

## Use

```
anine                              full-screen chat (type /help)
anine "explain @app.py"            one-shot; @file attaches a file
cat error.log | anine "what went wrong?"
anine -m llama3.1:8b               pick a model;  anine --set-default MODEL  remembers it
```

It works with most Ollama models. Models without native tool calling (or that handle it badly) are detected
automatically and driven with simple text action blocks instead (`--tools text|native|auto`).

## Safety

- Code and shell commands run inside a bubblewrap sandbox: no network by default, your home directory read-only
  (`--home off|ro|rw`), and keys, `.env` files, browser data and shell histories hidden.
- anine asks before running code or writing files (`-y` skips this; sudo/rm -rf/force-push style commands
  are always confirmed when unsandboxed or when home is writable).
- Every action is logged to `~/.anine/audit.log`; `/undo` reverts the last file change; backups are in `~/.anine/backups`.
- The model is told to refuse malware and attacks on systems you don't own, and to treat file contents as data, not instructions.
- Without `bwrap` (or with `--no-sandbox`) code runs with your full user privileges. Local models make mistakes: read what you approve.

Customize behavior with `/rules <text>` (persist with `/config save`) or an `ANINE.md` file in your project folder.

## License

MIT
