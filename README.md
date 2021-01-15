# blog

This repo contains my personal blog, and a simple/stupid "framework" (nope) to help me manage it.

## Usage

- write your posts in Markdown under `blog/posts/src`
- run `make` to generate HTML from Markdown
- run `py rss_generator.py` to generate RSS Feed, `blog/atom.xml`
- run `py sitemap_generator.py` to generate sitemap, `blog/sitemap.xml`
- run `make sync` to sync the blog to server

## Misc

Markdown-to-HTML transformation relies on Pandoc, template and metadata of generated HTML can be found at `blog/posts/templates` 

Enjoy

---

P.S. visit my blog at http://yanjieze.xyz (mostly in Chinese)
