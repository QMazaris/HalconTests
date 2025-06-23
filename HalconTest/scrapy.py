#!/usr/bin/env python3
"""
halcon_discover_ops.py

Enumerate all HALCON operator pages for
    • HALCON 12 classic reference
    • HALCON 19.05 Deep Learning add-on

Outputs CSV:  name, url, discovered_from  (no DB writes)

Usage:
    python halcon_discover_ops.py            # minimal output
    python halcon_discover_ops.py -v         # very verbose
"""
from __future__ import annotations
import argparse, csv, os, re, sys, time
from collections import defaultdict, deque
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

ROOTS = {
    "h12":    "https://www.mvtec.com/doc/halcon/12/en/index.html",
    "dl1905": "https://www.mvtec.com/doc/halcon/1905/en/toc_deeplearning.html",
}
UA       = {"User-Agent": "HalconScraper/0.3 (+https://example.com)"}
SLEEP    = 0.15               # throttle
OP_PAT   = re.compile(r"^[a-z0-9_]+\.html$")     # file.html typical of operators

# ------------------------------------------------------------------------------
def log(msg: str, level: str="INFO", verbose: bool=False):
    if verbose:
        print(f"[{level}] {msg}", flush=True)

def fetch(url: str, verbose: bool=False) -> BeautifulSoup:
    log(f"GET {url}", "HTTP", verbose)
    time.sleep(SLEEP)
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def frame_srcs(soup: BeautifulSoup, base: str) -> list[str]:
    """Return absolute src URLs of any <frame> or <iframe> tags."""
    out=[]
    for fr in soup.find_all(["frame", "iframe"]):
        if fr.has_attr("src"):
            out.append(urljoin(base, fr["src"]))
    return out

def discover(root_name: str, entry_url: str, verbose: bool=False) -> dict[str,str]:
    """
    BFS crawl limited to the same hostname; collect operator pages.
    Returns  {operator_name: absolute_url}
    """
    parsed_root = urlparse(entry_url)
    same_host   = f"{parsed_root.scheme}://{parsed_root.netloc}"

    queue: deque[str] = deque([entry_url])
    seen:  set[str]   = set()
    ops:   dict[str,str] = {}

    while queue:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        try:
            soup = fetch(url, verbose)
        except Exception as e:
            log(f"!! fetch failed: {e}", "WARN", verbose)
            continue

        # 1) Dive into framesets first
        for fsrc in frame_srcs(soup, url):
            if fsrc.startswith(same_host):
                queue.append(fsrc)

        # 2) Examine every <a href>
        for a in soup.find_all("a", href=True):
            href = urljoin(url, urldefrag(a["href"])[0])
            if not href.startswith(same_host):
                continue

            # enqueue more pages to inspect
            if "toc_" in href and href not in seen:
                queue.append(href)

            # operator-candidate?
            path = urlparse(href).path
            filename = os.path.basename(path)
            if "/operators/" in path or OP_PAT.match(filename):
                name = os.path.splitext(filename.lower())[0]
                if name and name not in ops:
                    ops[name] = href
                    log(f"    + op {name}", "OP", verbose)

    log(f"{root_name}: discovered {len(ops)} ops, visited {len(seen)} pages",
        "STAT", verbose)
    return ops

# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every page fetch & operator found")
    args = ap.parse_args()

    all_ops = defaultdict(dict)
    for key, url in ROOTS.items():
        all_ops[key] = discover(key, url, args.verbose)

    # ---- summary ----
    print("\n=== DISCOVERY SUMMARY ===")
    for k,v in all_ops.items():
        print(f"{k:7s}: {len(v):4d} operators")

    # ---- CSV export ----
    with open("halcon_ops.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "url", "version"])
        for ver, items in all_ops.items():
            for n,u in items.items():
                w.writerow([n,u,ver])

    print("\nCSV written to halcon_ops.csv")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
