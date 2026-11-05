"""Read report links from Git blobs; never import or extract student files."""

from html.parser import HTMLParser
import posixpath
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "img":
            self.sources.extend(value for key, value in attrs if key.lower() == "src" and value)


def report_links(source):
    def walk(tokens):
        for token in tokens:
            if token.type in ("image", "link_open"):
                yield token.attrGet("src" if token.type == "image" else "href")
            if token.type in ("html_block", "html_inline"):
                parser = Images()
                parser.feed(token.content)
                yield from parser.sources
            if token.children:
                yield from walk(token.children)
    return set(walk(MarkdownIt("commonmark").parse(source)))


def local_path(report, address):
    """None means external/anchor; ValueError means outside the repository."""
    parsed = urlsplit(address)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    if any(c in path for c in "\\\x00\r\n"):
        raise ValueError("Invalid local path")
    target = posixpath.normpath(path.lstrip("/") if path.startswith("/") else
                              posixpath.join(posixpath.dirname(report), path))
    if target == ".." or target.startswith("../"):
        raise ValueError("Path escapes repository")
    return target
