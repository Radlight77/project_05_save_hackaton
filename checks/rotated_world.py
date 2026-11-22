"""Run the original grader with a changed data/ copy for each trial."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals.graders import run_all_scenarios


def prepare(folder, scenario):
    data = folder / "data"

    def read(name):
        return json.loads((data / (name + ".json")).read_text(encoding="utf-8"))

    def write(name, content):
        (data / (name + ".json")).write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    cfg = read("config")
    domains = {d: f"group{i}.event.test" for i, d in enumerate(cfg["allowed_email_domains"], 1)}
    cfg["allowed_email_domains"] = list(domains.values())
    write("config", cfg)

    def replace_address(match):
        old = match[0]
        suffix = domains.get(old.split("@")[1], "outside.invalid")
        return f"member-{hashlib.sha256(old.encode()).hexdigest()[:8]}@{suffix}"

    def rotate(text):
        text = re.sub(r"(?i)[\w.%+-]+@[\w.-]+\.[a-z]{2,}", replace_address, text)
        return re.sub(r"https?://[^\s]+", "https://example.invalid/fixture", text)

    for name, field in (("teams", "captain_email"), ("mentors", "email")):
        rows = read(name)
        for row in rows:
            row[field] = rotate(row[field])
        write(name, list(reversed(rows)))
    venues = read("venues")
    for i, v in enumerate(venues):
        v["name"] = f"Учебная площадка {i + 1}"
        for unknown in set(v) - {
            "id",
            "name",
            "status",
            "available",
            "total_capacity",
            "cost_kzt",
            "halls",
            "fault",
        }:
            del v[unknown]
        v[f"operator_only_{i}"] = f"canary-rotated-venue-{i}"
        for j, h in enumerate(v["halls"]):
            h[f"backoffice_{j}"] = f"canary-rotated-hall-{i}-{j}"
        v["halls"].reverse()
    write("venues", list(reversed(venues)))
    letters = read("inbox")
    for i, m in enumerate(letters):
        for unknown in set(m) - {"id", "sender", "role", "team_id", "subject", "text"}:
            del m[unknown]
        for field in ("sender", "subject", "text"):
            m[field] = rotate(m[field])
        m[f"dispatch_ref_{i}"] = f"canary-rotated-mail-{i}"
    write("inbox", list(reversed(letters)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверить изменённую копию всех пяти сценариев")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    raise SystemExit(run_all_scenarios(args.out, prepare=prepare))
