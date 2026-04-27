# Memory restore — for the other machine

Copy this entire `.claude/memory/` directory back into Claude's per-project
auto-memory location:

```bash
# After git clone + checkout preprod_v03:
PROJECT_HASH_DIR="$HOME/.claude/projects/-home-ubuntu-PycharmProjects-DocWain/memory/"
mkdir -p "$PROJECT_HASH_DIR"
cp -r .claude/memory/* "$PROJECT_HASH_DIR"
```

If your project working-directory hash differs (Claude derives the path
from the absolute working dir), put the files in the path Claude uses on
that machine — the project-hash directory under `~/.claude/projects/`.

Files here are user-authored project memories (preferences, project
direction, feedback). They have no executable code.
