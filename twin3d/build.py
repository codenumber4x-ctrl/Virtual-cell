"""Embed data.json into template.html -> index.html."""
import pathlib
d = pathlib.Path(__file__).parent
(d / "index.html").write_text((d / "template.html").read_text().replace("__DATA__", (d / "data.json").read_text()))
