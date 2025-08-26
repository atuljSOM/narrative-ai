
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

def render_briefing(template_dir: str, out_path: str, brand: str, aligned, outputs):
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=select_autoescape())
    tpl = env.get_template("briefing.html.j2")
    html = tpl.render(brand=brand, aligned=aligned, outputs=outputs)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path
