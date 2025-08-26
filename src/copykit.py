
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def render_copy_for_actions(template_dir: str, out_assets_dir: str, actions: list[dict]) -> dict[str, list[str]]:
    out = {}
    env = Environment(loader=FileSystemLoader(template_dir))
    Path(out_assets_dir).mkdir(parents=True, exist_ok=True)
    for a in actions:
        play_id = a.get("play_id", a.get("id","action"))
        files = []
        for tmpl in (a.get("copy_snippets") or []):
            try:
                t = env.get_template(f"copy/{tmpl}")
            except Exception:
                continue
            fname = f"{play_id}_{tmpl.replace('.j2','').replace('.txt','')}.txt"
            path = Path(out_assets_dir)/fname
            txt = t.render(brand=a.get("brand",""), action=a)
            path.write_text(txt, encoding="utf-8")
            files.append(str(path))
        out[play_id] = files
    return out
