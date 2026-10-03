"""Use native MkDocs navigation for article API links and an editorial-only TOC."""

from mkdocs.structure.nav import Link, Section


def remove_api_toc_entries(toc, roots):
    def keep(items):
        result = []
        for item in items:
            children = keep(item.children)
            if any(
                item.id == root
                or item.id.startswith(root + ".")
                or item.id.startswith(root + "--")
                for root in roots
            ):
                result.extend(children)
            else:
                item.children = children
                result.append(item)
        return result

    toc.items = keep(toc.items)


def expand_article_navigation(nav, references, resolved, icons, api_kinds):
    defaults = icons.get("defaults", {})

    def set_icon(item, icon):
        # Material renders meta.icon natively for Pages, Sections and Links.
        item.meta = {**getattr(item, "meta", {}), "icon": icon}

    def visit(items, parent=None):
        result = []
        for item in items:
            if item.is_section:
                set_icon(
                    item, icons.get("sections", {}).get(item.title, defaults.get("section"))
                )
                item.children = visit(item.children, item)
            elif item.is_page:
                set_icon(
                    item,
                    icons.get("pages", {}).get(
                        item.file.src_uri, item.meta.get("icon", defaults.get("page"))
                    ),
                )
                names = list(dict.fromkeys(references.get(item.file.src_uri, [])))
                if len(names) > 1:
                    url = item.url or "./"
                    links = [Link("Overview", url)]
                    set_icon(links[0], defaults.get("overview"))
                    for name in names:
                        link = Link(name.rsplit(".", 1)[-1], url + "#" + resolved[name])
                        set_icon(link, defaults.get(api_kinds[name], defaults.get("page")))
                        links.append(link)
                    section = Section(item.title, links)
                    section.parent = parent
                    section.meta = item.meta
                    for link in links:
                        link.parent = section
                    # MkDocs activates a Page and its ancestors during rendering.
                    # Retain the original Page in nav.pages for ordering/search.
                    item.parent = section
                    item = section
            else:
                set_icon(item, defaults.get("page"))
            result.append(item)
        return result

    nav.items = visit(nav.items)
