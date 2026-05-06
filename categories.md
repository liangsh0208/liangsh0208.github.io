---
layout: default
title: 分类
permalink: /categories/
---

<div class="category-list">
{% for category in site.categories %}
  <h2 id="{{ category[0] }}">{{ category[0] }}</h2>
  <ul>
    {% for post in category[1] %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <small>{{ post.date | date: "%Y-%m-%d" }}</small>
    </li>
    {% endfor %}
  </ul>
{% endfor %}
</div>
