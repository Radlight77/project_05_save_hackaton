"""Regression checks for the policy boundary, independent from the original grader."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import mcp_server as server
import reset
import world
from run_store import save_state
from scheduling import RuleError, key, overlaps, plan, venue_rule


@pytest.fixture
def ready(tmp_path, monkeypatch):
    reset.reset_world(1, tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = world.load_config()
    teams, mentors = server.roster()
    for venue in server.catalog():
        try:
            venue_rule(venue, teams, cfg)
        except RuleError:
            continue
        break
    state = {
        "status": "running",
        "selected_venue": venue["id"],
        "decision": "approve",
        "approved_quote": {"venue_id": venue["id"], "cost_kzt": venue["cost_kzt"]},
    }
    save_state(state)
    booking = server.Booking(venue_id=venue["id"], idempotency_key=key("book", venue["id"]))
    return state, venue, teams, mentors, booking


def make_grid(ready):
    _, venue, teams, mentors, booking = ready
    assert server.book_venue(booking)["ok"]
    schedule = plan(venue, teams, mentors, [])
    items = [{**r, "idempotency_key": key("slot", r["team_id"])} for r in schedule.values()]
    assert server.assign_slots(server.Assignments(items=items))["ok"]
    return server.notification_drafts()["data"]


def test_approval_is_bound_to_price_and_venue(ready):
    state, venue, _, _, booking = ready
    for decision, price, vid in (
        (None, venue["cost_kzt"], venue["id"]),
        ("approve", venue["cost_kzt"] - 1, venue["id"]),
        ("approve", venue["cost_kzt"], "another"),
    ):
        state.update(decision=decision, approved_quote={"cost_kzt": price, "venue_id": vid})
        save_state(state)
        assert server.book_venue(booking)["code"] == "approval_required"
    assert not world.read_bookings()


def test_fresh_price_and_budget(ready):
    _, venue, _, _, booking = ready
    cfg = world.load_config()
    cfg["max_budget_kzt"] = venue["cost_kzt"] - 1
    Path("data/config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert server.book_venue(booking)["code"] == "over_budget"
    with pytest.raises(ValidationError):
        server.Booking(**booking.model_dump(), cost_kzt=0)
    assert world.read_bookings() == []


def test_capacity_and_key(ready):
    _, venue, _, _, booking = ready
    bad = booking.model_copy(update={"idempotency_key": "different"})
    assert server.book_venue(bad)["code"] == "invalid_key"
    p = Path("data/venues.json")
    rows = json.loads(p.read_text(encoding="utf-8"))
    next(v for v in rows if v["id"] == venue["id"])["total_capacity"] = 1
    p.write_text(json.dumps(rows), encoding="utf-8")
    assert server.book_venue(booking)["code"] == "capacity"
    assert not world.read_bookings()


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("recipient", "qa@outside.invalid", "recipient_domain"),
        ("subject", "https://example.invalid/", "unsafe_content"),
        ("text", "www.example.invalid", "unsafe_content"),
        ("text", "Прежнее расписание", "notification_mismatch"),
        ("idempotency_key", "changed", "invalid_key"),
    ],
)
def test_invalid_batch_has_no_partial_mail(ready, field, value, code):
    drafts = make_grid(ready)
    drafts[1][field] = value
    assert server.send_notifications(server.Messages(items=drafts))["code"] == code
    assert not world.read_outbox()
    assert server.verify_transfer()["code"] == "missing_notifications"


def test_a_replay_does_not_repeat_effects(ready):
    drafts = make_grid(ready)
    assert server.book_venue(ready[-1])["data"]["replayed"]
    request = server.Messages(items=drafts)
    assert server.send_notifications(request)["data"]["written"] == len(drafts)
    assert server.send_notifications(request)["data"]["replayed"] == len(drafts)
    assert server.verify_transfer()["ok"]
    p = Path("outbox.json")
    sent = world.read_outbox()
    sent[0]["text"] = "Время изменилось"
    p.write_text(json.dumps(sent), encoding="utf-8")
    assert server.verify_transfer()["code"] == "notification_mismatch"


def test_half_open_time_intervals():
    assert overlaps("11:00-11:30", "11:20-11:50")
    assert not overlaps("11:00-11:30", "11:30-12:00")


def test_compatible_captain_requests_both_fit(ready):
    _, venue, teams, mentors, _ = ready
    a, b = sorted(teams, key=lambda team: team["id"])[:2]
    mentor_time, pitch_time = "11:00-11:30", "16:00-16:15"
    inbox = [
        {"role": "captain", "sender": t["captain_email"], "team_id": t["id"], "text": slot}
        for t, slot in ((a, mentor_time), (b, pitch_time))
    ]
    grid = plan(venue, teams, mentors, inbox)
    assert set(grid) == {t["id"] for t in teams}
    assert (grid[a["id"]]["mentor_slot"], grid[b["id"]]["pitch_slot"]) == (mentor_time, pitch_time)


@pytest.mark.parametrize("tool, arguments", [
    ("book_venue", {"request": {"venue_id": "V-HUB"}}),
    ("read_roster", {"unexpected": "private test value"}),
])
def test_schema_rejections_give_a_public_receipt(ready, tool, arguments):
    reply = asyncio.run(server.mcp.call_tool(tool, arguments))
    assert reply.isError and reply.structuredContent == {"ok": False, "code": "invalid_arguments"}
    line = Path("journal.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    record = json.loads(line)
    assert (record["event"], record["tool"], record["code"]) == ("tool_rejected", tool, "invalid_arguments")
    assert "private test value" not in line
    assert not world.read_bookings() and not world.read_outbox()


def test_lookup_leaves_held_transfer_budget_and_checkpoint_intact(tmp_path, monkeypatch):
    import argparse
    from agent import execute, initial
    from run_store import read_state

    reset.reset_world(1, tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = world.load_config()
    state = initial()
    state.update(status="waiting_approval", checkpoint="book", turns=cfg["max_turns"] - 1,
                 attempts=1, skill_activated=True)
    save_state(state)
    args = argparse.Namespace(query=f"Когда встреча {world.load_teams()[0]['id']}?")
    assert asyncio.run(execute(args)) == 0
    saved = read_state()
    lookup = saved.pop("last_query")
    assert saved == state and lookup["turns"] == 2
    assert not lookup["skill_activated"] and lookup["token_usage"]["calls"] == 0


def test_query_returns_team_and_both_current_times(tmp_path):
    import subprocess
    import sys

    reset.reset_world(3, tmp_path)
    team = world.load_teams(tmp_path)[0]
    agent = Path(__file__).resolve().parents[1] / "agent.py"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(agent), "--query", f"Статус {team['id']}?"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=60,
    )
    for value in (
        team["id"],
        team["name"],
        team["current_mentor_slot"],
        team["current_pitch_slot"],
        "teams.json",
    ):
        assert value in result.stdout
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert not state["skill_activated"] and state["token_usage"]["calls_started"] == 0
    assert not any(
        (tmp_path / name).exists() for name in ("bookings.json", "schedule.json", "outbox.json")
    )
