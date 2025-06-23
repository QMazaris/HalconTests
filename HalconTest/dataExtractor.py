# -*- coding: utf-8 -*-
import os, re, time, requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag, NavigableString, FeatureNotFound

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


# -----------------------------------------------------------------------------
def parse_operator_page(url: str) -> dict[str, str]:
    """Return {name, signature, description, page_dump}."""
    time.sleep(DELAY)
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    soup = _soup(r.text)

    name = os.path.splitext(os.path.basename(urlparse(url).path))[0]

    # -------- Signature --------
    h_sig = soup.find(lambda t: isinstance(t, Tag)
                      and t.name == "h2"
                      and (
                          t.get("id") == "sec_synopsis"
                          or t.find("a", attrs={"name": "sec_synopsis"})
                      ))
    signature = _first_signature_block(h_sig) if h_sig else ""

    # -------- Description -------
    h_desc = soup.find(lambda t: isinstance(t, Tag)
                       and t.name == "h2"
                       and (
                           t.get("id") == "sec_description"
                           or t.find("a", attrs={"name": "sec_description"})
                       ))
    description = _collect_description(h_desc) if h_desc else ""

    page_dump = soup.get_text("\n", strip=True)

    return {
        "name":        name,
        "signature":   signature,
        "description": description,
        "page_dump":   page_dump,
    }


# -------------------- quick demo --------------------
if __name__ == "__main__":
    urls = [
        # classic
        "https://www.mvtec.com/doc/halcon/12/en/abs_diff_image.html",
        # deep-learning
        "https://www.mvtec.com/doc/halcon/1905/en/read_dl_classifier.html",
    ]
    for u in urls:
        rec = parse_operator_page(u)
        print(f"\n{rec['name']}")
        print("  signature :", rec['signature'])
        print("  desc      :", rec['description'][:120], "…")
        print("page_dump: ", rec["page_dump"])
