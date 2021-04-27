# blog: The Last Refugee

This repo contains Yanjie Ze's personal blog.

His research interest lies in computer vision, reinforcement learning and natural language processing. 

## Usage

- write your posts in Markdown under `blog/posts/src`

- run `make` to generate HTML from Markdown(no chinese character in file name!)

- run `py rss_generator.py` to generate RSS Feed, `blog/atom.xml`

- run `py sitemap_generator.py` to generate sitemap, `blog/sitemap.xml`

- insert a picture by:
```html
<center>
<img src="../imgs/x.jpg">
</center>

<center>
<img src="../imgs/x.png">
</center>
```

## Misc

Markdown-to-HTML transformation relies on Pandoc, template and metadata of generated HTML can be found at `blog/posts/templates` 

Enjoy：）

---

P.S. visit my blog at http://yanjieze.xyz (mostly in Chinese)
