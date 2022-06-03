#!/usr/bin/python3
'''
Entrypoint for the utils package.
'''
import base64
import binascii
import json
import os
import sys

import requests
import yaml


def _write(content: str, path: str) -> int:
    try:
        with open(f"/data/{path}", "w") as f:
            f.write(content)
        return 0
    except IOError as e:
        return e.errno
    except Exception:
        return 1


def _decode(s: str) -> str:
    s = s.replace("\n", "")
    b = base64.b64decode(s)
    return b.decode("utf-8")


def download(owner: str, repo: str, path_src: str, path_dst: str) -> int:
    """
    Download the file frome github

    Parameters
    ----------
    owner: `str`
    Repository owner.

    repo: `str`
    The repository name.

    path_src: `str`
    The file path in the repository
    
    path_dst: `str`
    The path where the file will be exported to the file system.
    
    Returns
    -------
    `int` Error code.
    """
    
    res = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path_src}?ref=main")
    if not res.ok:
        return 1
    try:
        data = res.json()
        decoded = _decode(data.get("content", ""))
        return _write(decoded, path_dst)
    except (json.JSONDecodeError, binascii.Error):
        return 1


def main():
    command = sys.argv[1]
    if command == "download":
        owner = os.environ.get("OWNER", None)
        repo = os.environ.get("REPO", None)
        path_src = os.environ.get("PATH_SRC", None)
        path_dst = os.environ.get("PATH_DST", None)
        errcode = download(owner, repo, path_src, path_dst)
        print(yaml.dump({"errcode": errcode}))


if __name__ == '__main__':
    main()
