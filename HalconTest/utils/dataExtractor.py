# -*- coding: utf-8 -*-
import os, re, time, requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag, FeatureNotFound
from typing import Iterable, Optional

UA, DELAY = {"User-Agent": "HalconScraper/0.8"}, 0.15
DUP_RE    = re.compile(r"\b(\w+)(?:\s+\1\b)+", flags=re.I)   # aa aa → aa


def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        return BeautifulSoup(html, "html.parser")


def _dedup(text: str) -> str:
    return DUP_RE.sub(r"\1", text)


def _first_signature_block(h2: Tag) -> str:
    """
    Given the <h2> 'Signature', return cleaned HDevelop prototype.
    Works for both
        <div name="hdevelop"> … <code>…</code>
    and
        <div data-if="hdevelop"> … <code>…</code>
    """
    sig_div = h2.find_next(
        lambda t: isinstance(t, Tag)
        and t.name == "div"
        and (
            t.get("name") == "hdevelop"
            or t.get("data-if") == "hdevelop"
        )
    )
    if not sig_div:
        return ""

    code = sig_div.find("code") or sig_div.find("pre") or sig_div
    txt  = code.get_text(" ", strip=True)
    txt  = re.sub(r"\s+", " ", txt)
    return txt


def _collect_description(desc_h2: Tag) -> str:
    """
    Gather <p> blocks (and list items) until we hit the next headline
    or the generic parameter list.
    """
    out = []
    for sib in desc_h2.next_siblings:
        if isinstance(sib, Tag) and sib.name.startswith("h"):
            break
        if isinstance(sib, Tag) and sib.name == "dl" and sib.get("class") == ["generic"]:
            break
        if isinstance(sib, Tag) and sib.name in ("p", "li"):
            t = sib.get_text(" ", strip=True)
            if t:
                out.append(_dedup(t))
    return " ".join(out)


def _collect_parameters_or_results(section_h2: Tag) -> str:
    """
    Collect parameter or result information from a section.
    Look for definition lists (dl) and paragraphs.
    """
    if not section_h2:
        return ""
    
    out = []
    for sib in section_h2.next_siblings:
        if isinstance(sib, Tag) and sib.name.startswith("h"):
            break
        if isinstance(sib, Tag):
            if sib.name == "dl":
                # Handle definition lists (parameter descriptions)
                for dt in sib.find_all("dt"):
                    dd = dt.find_next_sibling("dd")
                    param_name = dt.get_text(" ", strip=True)
                    param_desc = dd.get_text(" ", strip=True) if dd else ""
                    if param_name:
                        out.append(f"{param_name}: {param_desc}")
            elif sib.name in ("p", "li", "div"):
                t = sib.get_text(" ", strip=True)
                if t:
                    out.append(_dedup(t))
            elif sib.name == "table":
                # Handle parameter tables
                for row in sib.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        param_name = cells[0].get_text(" ", strip=True)
                        param_desc = " ".join(cell.get_text(" ", strip=True) for cell in cells[1:])
                        if param_name and not param_name.lower() in ["parameter", "name"]:
                            out.append(f"{param_name}: {param_desc}")
    return " ".join(out)


# ------------------------------------------------------------
# Generic helpers

def _find_section(
    soup: BeautifulSoup,
    *,
    ids: Iterable[str] = (),
    anchor_names: Iterable[str] = (),
    keywords: Iterable[str] = (),
) -> Optional[Tag]:
    """Return the first <h2>/<h3> tag matching any of the given criteria."""
    kw_lower = {k.lower() for k in keywords}
    return soup.find(
        lambda t: isinstance(t, Tag)
        and t.name in ("h2", "h3")
        and (
            (t.get("id") and t.get("id") in ids)
            or any(t.find("a", attrs={"name": n}) is not None for n in anchor_names)
            or any(k in t.get_text().lower() for k in kw_lower)
        )
    )


def _extract_signature(soup: BeautifulSoup) -> str:
    h_sig = _find_section(
        soup,
        ids=("sec_synopsis",),
        anchor_names=("sec_synopsis",),
        keywords=("synopsis",),
    )
    return _first_signature_block(h_sig) if h_sig else ""


def _extract_description(soup: BeautifulSoup) -> str:
    h_desc = _find_section(
        soup,
        ids=("sec_description",),
        anchor_names=("sec_description",),
        keywords=("description",),
    )
    return _collect_description(h_desc) if h_desc else ""


def _extract_parameters(soup: BeautifulSoup) -> str:
    h_params = _find_section(
        soup,
        ids=("sec_parameters",),
        anchor_names=("sec_parameters",),
        keywords=("parameter", "input"),
    )
    return _collect_parameters_or_results(h_params) if h_params else ""


def _extract_results(soup: BeautifulSoup) -> str:
    h_res = _find_section(
        soup,
        ids=("sec_result",),
        anchor_names=("sec_result",),
        keywords=("result", "output", "return"),
    )
    return _collect_parameters_or_results(h_res) if h_res else ""


def parse_operator_page(url: str) -> dict[str, str]:
    """Return {name, signature, description, parameters, results}."""
    time.sleep(DELAY)
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    soup = _soup(r.text)

    name = os.path.splitext(os.path.basename(urlparse(url).path))[0]

    return {
        "name":        name,
        "signature":   _extract_signature(soup),
        "description": _extract_description(soup),
        "parameters":  _extract_parameters(soup),
        "results":     _extract_results(soup),
    }


# -------------------- quick demo --------------------
if __name__ == "__main__":
    urls = [
        # classic
        "https://www.mvtec.com/doc/halcon/12/en/abs_diff_image.html",
        # deep-learning
        "https://www.mvtec.com/doc/halcon/1905/en/read_dl_classifier.html",
        # With signatures
        "https://www.mvtec.com/doc/halcon/1811/en/get_object_model_3d_params.html",

    ]
    for u in urls:
        rec = parse_operator_page(u)
        print(f"\n{rec['name']}")
        print("  signature  :", rec['signature'])
        print("  desc       :", rec['description'][:100] + "..." if len(rec['description']) > 100 else rec['description'])
        print("  parameters :", rec['parameters'][:100] + "..." if len(rec['parameters']) > 100 else rec['parameters'])
        print("  results    :", rec['results'][:100] + "..." if len(rec['results']) > 100 else rec['results'])

