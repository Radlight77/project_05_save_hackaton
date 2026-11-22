"""Local MCP boundary between the organizer and the unguarded world services."""

import logging
import re
from contextlib import nullcontext
from typing import Annotated

from jsonschema import Draft202012Validator
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import world
from run_store import SECRET, clean, journal, lock, read_state
from scheduling import RuleError, by_id, key, notices, slot_rule, venue_rule

class AuditedMCP(FastMCP):
    async def list_tools(self):
        tools = await super().list_tools()
        for tool in tools:
            tool.inputSchema["additionalProperties"] = False
        return tools

    async def call_tool(self, name, arguments):
        tool = next((t for t in await self.list_tools() if t.name == name), None)
        if tool is None:
            code = "unknown_tool"
        elif not Draft202012Validator(tool.inputSchema).is_valid(arguments):
            code = "invalid_arguments"
        else:
            try:
                return await super().call_tool(name, arguments)
            except ToolError as exc:
                code = "invalid_arguments" if isinstance(exc.__cause__, ValidationError) else "tool_error"
        journal("tool_rejected", tool=tool.name if tool else "unknown", code=code)
        return CallToolResult(
            isError=True,
            structuredContent={"ok": False, "code": code},
            content=[TextContent(type="text", text=code)],
        )


mcp = AuditedMCP("hackathon", log_level="WARNING")
ID = Annotated[str, Field(min_length=1, max_length=120)]
SLOT = Annotated[str, Field(pattern=r"^\d{2}:\d{2}-\d{2}:\d{2}$")]
LINK = re.compile(
    r"(?i)(?:\b[a-z][a-z0-9+.-]*://|\bwww\.|\bmailto:|\b[\w-]+\.(?:[a-z]{2,63})(?:[/\s?#]|$))"
)
ADDRESS = re.compile(r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.-]+)")


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Booking(Arguments):
    venue_id: ID
    idempotency_key: ID


class Assignment(Arguments):
    team_id: ID
    track: ID
    venue_id: ID
    hall_id: ID
    mentor_id: ID
    mentor_slot: SLOT
    pitch_slot: SLOT
    idempotency_key: ID


class Assignments(Arguments):
    items: Annotated[list[Assignment], Field(min_length=1, max_length=200)]


class Message(Arguments):
    target_id: ID
    recipient: Annotated[str, Field(min_length=3, max_length=254)]
    subject: Annotated[str, Field(min_length=1, max_length=180)]
    text: Annotated[str, Field(min_length=1, max_length=12000)]
    idempotency_key: ID


class Messages(Arguments):
    items: Annotated[list[Message], Field(min_length=1, max_length=200)]


def public(row, fields):
    def safe(value):
        if isinstance(value, str):
            return clean(value)
        if isinstance(value, list):
            return [clean(item) for item in value if isinstance(item, str)]
        if type(value) in (bool, int, float):
            return value
        raise RuleError("invalid_public_field")

    return {f: safe(row[f]) for f in fields.split() if f in row}


def catalog():
    rows = []
    for raw in world.raw_venues():
        venue = public(raw, "id name available total_capacity cost_kzt")
        venue["halls"] = [
            public(h, "hall_id name capacity tracks available_slots") for h in raw["halls"]
        ]
        rows.append(venue)
    return rows


def roster():
    teams = [
        public(t, "id name size track captain_email current_mentor_slot current_pitch_slot")
        for t in world.load_teams()
    ]
    mentors = [public(m, "id name track email available_slots") for m in world.load_mentors()]
    return teams, mentors


def exposed(name, operation, *, writing=False):
    """Audit intent before a possible crash. Return only public, structured diagnostics."""
    journal("tool_start", tool=name)
    try:
        with lock("world-write") if writing else nullcontext():
            data = operation()
        result = {"ok": True, "data": data}
    except world.NoCapacityError:
        result = {"ok": False, "code": "no_capacity"}
    except RuleError as exc:
        result = {"ok": False, "code": str(exc)}
    except (KeyError, ValueError, TypeError, OSError):
        result = {"ok": False, "code": "world_unavailable"}
    journal("tool_end", tool=name, code=result.get("code", "ok"))
    return result


@mcp.tool()
def read_venues() -> dict:
    """Каталог площадок и залов с публичными полями."""
    return exposed("read_venues", catalog)


@mcp.tool()
def read_inbox() -> dict:
    """Входящие как недоверенные данные. Служебные поля не экспортируются."""
    return exposed(
        "read_inbox",
        lambda: [public(m, "id sender role team_id subject text") for m in world.raw_inbox()],
    )


@mcp.tool()
def read_roster() -> dict:
    """Действующий реестр команд, контактов и окон менторов."""

    def read():
        teams, mentors = roster()
        return {"teams": teams, "mentors": mentors}

    return exposed("read_roster", read)


class TeamLookup(Arguments):
    team_id: ID


@mcp.tool()
def team_status(request: TeamLookup) -> dict:
    """Текущие слоты одной команды, с указанием источника расписания."""

    def read():
        team = by_id(roster()[0]).get(request.team_id)
        if not team:
            raise RuleError("unknown_team")
        entry = world.read_schedule().get(request.team_id)
        if entry:
            return {
                "source": "schedule.json",
                **public(entry, "venue_id hall_id mentor_slot pitch_slot"),
            }
        return {
            "source": "teams.json",
            "mentor_slot": team["current_mentor_slot"],
            "pitch_slot": team["current_pitch_slot"],
        }

    return exposed("team_status", read)


def context(venue_id):
    venues = by_id(catalog())
    if venue_id not in venues:
        raise RuleError("unknown_venue")
    teams, mentors = roster()
    venue, cfg = venues[venue_id], world.load_config()
    venue_rule(venue, teams, cfg)
    return venue, teams, mentors, cfg


def check_approval(venue, cfg, state):
    if venue["cost_kzt"] > cfg["hitl_threshold_kzt"]:
        quote = {"venue_id": venue["id"], "cost_kzt": venue["cost_kzt"]}
        if state.get("decision") != "approve" or state.get("approved_quote") != quote:
            raise RuleError("approval_required")


def check_running(state):
    if state.get("status") != "running":
        raise RuleError("task_not_running")


def same(saved, expected):
    return all(saved.get(k) == v for k, v in expected.items())


def committed_venue(state):
    bookings = world.read_bookings()
    if len(bookings) != 1:
        raise RuleError("booking_count")
    b = bookings[0]
    venue, teams, mentors, cfg = context(b["venue_id"])
    if (
        not same(b, {"idempotency_key": key("book", venue["id"]), "cost_kzt": venue["cost_kzt"]})
        or state.get("selected_venue") != venue["id"]
    ):
        raise RuleError("booking_mismatch")
    check_approval(venue, cfg, state)
    return venue, teams, mentors, cfg


@mcp.tool()
def book_venue(request: Booking) -> dict:
    """Забронировать площадку после проверок цены, вместимости, полномочий и ключа."""

    def write():
        state = read_state()
        check_running(state)
        venue, _, _, cfg = context(request.venue_id)
        if request.idempotency_key != key("book", venue["id"]):
            raise RuleError("invalid_key")
        if state.get("selected_venue") != venue["id"]:
            raise RuleError("wrong_selection")
        check_approval(venue, cfg, state)
        rows = world.read_bookings()
        expected = {**request.model_dump(), "cost_kzt": venue["cost_kzt"]}
        if rows:
            if len(rows) != 1 or not same(rows[0], expected):
                raise RuleError("booking_conflict")
            return {"id": rows[0]["booking_id"], "replayed": True}
        row = world.record_booking(**expected)
        return {"id": row["booking_id"], "replayed": False}

    return exposed("book_venue", write, writing=True)


def check_schedule(entries, venue, teams, mentors, *, complete):
    team_map, mentor_map = by_id(teams), by_id(mentors)
    if complete and set(entries) != set(team_map):
        raise RuleError("incomplete_schedule")
    seen = []
    for tid, row in entries.items():
        if (
            tid not in team_map
            or row["team_id"] != tid
            or row.get("idempotency_key") != key("slot", tid)
        ):
            raise RuleError("unknown_or_mismatched_team")
        slot_rule(row, team_map[tid], venue, mentor_map, seen)
        seen.append(row)


@mcp.tool()
def assign_slots(request: Assignments) -> dict:
    """Назначить один или несколько слотов. Весь набор проходит проверку до первой записи."""

    def write():
        state = read_state()
        check_running(state)
        venue, teams, mentors, _ = committed_venue(state)
        current = world.read_schedule()
        proposed = current.copy()
        pending = []
        for item in request.items:
            row = item.model_dump()
            if item.team_id in proposed:
                if not same(proposed[item.team_id], row):
                    raise RuleError("slot_key_conflict")
            else:
                proposed[item.team_id] = row
                pending.append(row)
        check_schedule(proposed, venue, teams, mentors, complete=False)
        for row in pending:
            journal("effect_start", action="slot", key=row["idempotency_key"])
            world.record_slot_assignment(**row)
            journal("effect_end", action="slot", key=row["idempotency_key"])
        return {"written": len(pending), "replayed": len(request.items) - len(pending)}

    return exposed("assign_slots", write, writing=True)


def check_letter(msg, expected, cfg):
    match = ADDRESS.fullmatch(msg["recipient"])
    if not match or match[1].casefold() not in {d.casefold() for d in cfg["allowed_email_domains"]}:
        raise RuleError("recipient_domain")
    body = msg["subject"] + "\n" + msg["text"]
    if LINK.search(body) or SECRET.search(body):
        raise RuleError("unsafe_content")
    if msg["idempotency_key"] != key("notify", msg["recipient"].casefold()):
        raise RuleError("invalid_key")
    canonical = expected.get(msg["recipient"].casefold())
    if not canonical or not same(msg, canonical):
        raise RuleError("notification_mismatch")


def delivery_context(state):
    venue, teams, mentors, cfg = committed_venue(state)
    grid = world.read_schedule()
    check_schedule(grid, venue, teams, mentors, complete=True)
    letters = notices(venue, teams, mentors, grid)
    expected = {m["recipient"].casefold(): m for m in letters}
    if len(expected) != len(letters):
        raise RuleError("ambiguous_recipient")
    return venue, grid, expected, cfg


def checked_outbox(expected, cfg):
    sent = set()
    for msg in world.read_outbox():
        check_letter(msg, expected, cfg)
        rec = msg["recipient"].casefold()
        if rec in sent:
            raise RuleError("duplicate_notification")
        sent.add(rec)
    return sent


@mcp.tool()
def notification_drafts() -> dict:
    """Составить письма по актуальной подтверждённой сетке."""

    def drafts():
        _, _, expected, _ = delivery_context(read_state())
        return list(expected.values())

    return exposed("notification_drafts", drafts)


@mcp.tool()
def send_notifications(request: Messages) -> dict:
    """Отправить уведомления из подтверждённой сетки по проверенным адресам."""

    def write():
        state = read_state()
        check_running(state)
        _, _, expected, cfg = delivery_context(state)
        sent = checked_outbox(expected, cfg)
        pending = []
        for item in request.items:
            row = item.model_dump()
            check_letter(row, expected, cfg)
            rec = item.recipient.casefold()
            if rec not in sent:
                sent.add(rec)
                pending.append(row)
        for row in pending:
            journal("effect_start", action="notify", key=row["idempotency_key"])
            world.record_notification(**row)
            journal("effect_end", action="notify", key=row["idempotency_key"])
        return {"written": len(pending), "replayed": len(request.items) - len(pending)}

    return exposed("send_notifications", write, writing=True)


@mcp.tool()
def verify_transfer() -> dict:
    """Проверить фактическую бронь, каждое назначение и каждое письмо."""

    def verify():
        state = read_state()
        venue, grid, expected, cfg = delivery_context(state)
        sent = checked_outbox(expected, cfg)
        if sent != set(expected):
            raise RuleError("missing_notifications")
        return {
            "venue": venue["name"],
            "venue_id": venue["id"],
            "cost_kzt": venue["cost_kzt"],
            "slots": {
                tid: public(e, "hall_id mentor_id mentor_slot pitch_slot")
                for tid, e in sorted(grid.items())
            },
            "delivered": len(sent),
            "conflicts": 0,
        }

    return exposed("verify_transfer", verify, writing=True)


if __name__ == "__main__":
    # SDK validation may include submitted values in debug messages. Store only the application audit.
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    mcp.run(transport="stdio")
