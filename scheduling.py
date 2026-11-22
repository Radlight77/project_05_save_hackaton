"""Pure scheduling rules shared by the planner and the tools."""

import hashlib
import json
import re
from itertools import product


class RuleError(ValueError):
    """A short public reason code; never put raw external data in it."""


def key(kind: str, identity: str) -> str:
    value = json.dumps([kind, identity], ensure_ascii=False, separators=(",", ":"))
    return kind + "-" + hashlib.sha256(value.encode()).hexdigest()[:32]


def by_id(rows):
    return {r["id"]: r for r in rows}


def interval(s: str) -> tuple[int, int]:
    match = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)", s)
    if not match:
        raise RuleError("invalid_interval")
    h, m, hh, mm = map(int, match.groups())
    lo, hi = 60 * h + m, 60 * hh + mm
    if lo >= hi:
        raise RuleError("invalid_interval")
    return lo, hi


def overlaps(a: str, b: str) -> bool:
    left, right = interval(a), interval(b)
    return left[0] < right[1] and right[0] < left[1]


def venue_rule(venue, teams, cfg):
    if not venue["available"]:
        raise RuleError("venue_unavailable")
    if venue["total_capacity"] < sum(t["size"] for t in teams):
        raise RuleError("capacity")
    if not 0 <= venue["cost_kzt"] <= cfg["max_budget_kzt"]:
        raise RuleError("over_budget")
    tracks = {t for h in venue["halls"] for t in h["tracks"]}
    if not set(cfg["required_tracks"]).union(t["track"] for t in teams) <= tracks:
        raise RuleError("missing_track")


def slot_rule(entry, team, venue, mentors, others):
    """Check intervals, resources and event type against their original sources."""
    hall = next((h for h in venue["halls"] if h["hall_id"] == entry["hall_id"]), None)
    mentor = mentors.get(entry["mentor_id"])
    if not hall or not mentor or entry["venue_id"] != venue["id"]:
        raise RuleError("unknown_resource")
    track = team["track"]
    if entry["track"] != track or track not in hall["tracks"] or mentor["track"] != track:
        raise RuleError("track_mismatch")
    if hall["capacity"] < team["size"]:
        raise RuleError("hall_capacity")
    ms, ps = entry["mentor_slot"], entry["pitch_slot"]
    if ms not in mentor["available_slots"] or not {ms, ps} <= set(hall["available_slots"]):
        raise RuleError("unavailable_slot")
    start, end = interval(ms)
    pitch_start, pitch_end = interval(ps)
    old_ms, old_ps = interval(team["current_mentor_slot"]), interval(team["current_pitch_slot"])
    if (
        end - start != old_ms[1] - old_ms[0]
        or pitch_end - pitch_start != old_ps[1] - old_ps[0]
        or end > pitch_start
    ):
        raise RuleError("wrong_session_time")
    for other in others:
        if other["hall_id"] == entry["hall_id"] and any(
            overlaps(a, b) for a, b in product((ms, ps), (other["mentor_slot"], other["pitch_slot"]))
        ):
            raise RuleError("hall_overlap")
        if other["mentor_id"] == entry["mentor_id"] and overlaps(ms, other["mentor_slot"]):
            raise RuleError("mentor_overlap")


def preferences(inbox, teams):
    wanted = {}
    for letter in inbox:
        team = teams.get(letter.get("team_id"))
        if (
            team
            and letter["role"] == "captain"
            and letter["sender"].casefold() == team["captain_email"].casefold()
        ):
            ranges = re.findall(r"\b\d\d:\d\d-\d\d:\d\d\b", letter["text"])
            wanted.setdefault(team["id"], set()).update(ranges)
    return wanted


def plan(venue, teams, mentors, inbox):
    mentor_by_id = by_id(mentors)
    wanted = preferences(inbox, by_id(teams))
    options = {}
    for team in teams:
        choices = []
        for hall, mentor in product(venue["halls"], mentors):
            for ms, ps in product(
                sorted(set(mentor["available_slots"]) & set(hall["available_slots"])),
                sorted(hall["available_slots"]),
            ):
                entry = {
                    "team_id": team["id"],
                    "track": team["track"],
                    "venue_id": venue["id"],
                    "hall_id": hall["hall_id"],
                    "mentor_id": mentor["id"],
                    "mentor_slot": ms,
                    "pitch_slot": ps,
                }
                try:
                    slot_rule(entry, team, venue, mentor_by_id, ())
                except RuleError:
                    continue
                choices.append(entry)
        wish = wanted.get(team["id"], set())
        options[team["id"]] = sorted(
            choices,
            key=lambda e: (
                -len(wish & {e["mentor_slot"], e["pitch_slot"]}),
                e["mentor_slot"],
                e["pitch_slot"],
                e["hall_id"],
                e["mentor_id"],
            ),
        )
    order = sorted(teams, key=lambda t: (not wanted.get(t["id"]), len(options[t["id"]]), t["id"]))
    # Upper bound on wishes still satisfiable, ignoring collisions between teams.
    def score(entry):
        return len(wanted.get(entry["team_id"], set()) & {entry["mentor_slot"], entry["pitch_slot"]})

    remaining = [0] * (len(order) + 1)
    for i in reversed(range(len(order))):
        choices = options[order[i]["id"]]
        if not choices:
            raise RuleError("no_complete_schedule")
        remaining[i] = remaining[i + 1] + score(choices[0])

    assigned = {}
    best = None
    best_score = -1

    def place(index, achieved):
        nonlocal best, best_score
        if index == len(order):
            best, best_score = assigned.copy(), achieved
            return
        team = order[index]
        for e in options[team["id"]]:
            candidate_score = achieved + score(e)
            if candidate_score + remaining[index + 1] <= best_score:
                break  # Later options for this team satisfy at most as many wishes.
            try:
                slot_rule(e, team, venue, mentor_by_id, assigned.values())
            except RuleError:
                continue
            assigned[team["id"]] = e
            place(index + 1, candidate_score)
            del assigned[team["id"]]

    place(0, 0)
    if best is None:
        raise RuleError("no_complete_schedule")
    return best


def notices(venue, teams, mentors, schedule):
    """Build the eight (or roster-sized) letters from the recorded timetable."""
    if set(schedule) != set(by_id(teams)):
        raise RuleError("incomplete_schedule")
    hall_names = {h["hall_id"]: h["name"] for h in venue["halls"]}
    people = by_id(mentors)
    mails = []

    def add(who, address, text):
        mails.append(
            {
                "target_id": who,
                "recipient": address,
                "subject": "Хакатон перенесён",
                "text": text,
                "idempotency_key": key("notify", address.casefold()),
            }
        )

    for team in sorted(teams, key=lambda t: t["id"]):
        e = schedule[team["id"]]
        text = (
            f"{team['name']}, ваша новая площадка — {venue['name']}. {hall_names[e['hall_id']]}. "
            f"Встреча с ментором {people[e['mentor_id']]['name']} — {e['mentor_slot']}. "
            f"Выступление — {e['pitch_slot']}."
        )
        add(team["id"], team["captain_email"], text)
    names = {t["id"]: t["name"] for t in teams}
    for mentor in sorted(mentors, key=lambda m: m["id"]):
        rows = sorted(
            (e for e in schedule.values() if e["mentor_id"] == mentor["id"]),
            key=lambda e: e["mentor_slot"],
        )
        lines = [
            f"{names[e['team_id']]} — {hall_names[e['hall_id']]}, встреча {e['mentor_slot']}, питч {e['pitch_slot']}."
            for e in rows
        ]
        add(
            mentor["id"],
            mentor["email"],
            f"{mentor['name']}, хакатон пройдёт на площадке {venue['name']}. " + " ".join(lines),
        )
    return mails
