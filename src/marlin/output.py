"""Output discipline: --json gates everything; auto-JSON when stdout is piped.

Agents parse stdout. Humans get rich tables. One switch, checked everywhere.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.theme import Theme

# NemoStation brand palette (company/brand.md). marlin = coral; hero numbers =
# splash orange; CTA/links = accent red; warm = amber; secondary = inkLight.
# No green/blue/yellow — they're off-palette.
BRAND = Theme(
    {
        "model": "#E76F57",  # marlinCoral — model/CLI name + accents
        "accent": "#E76F57",
        "ok": "bold #FF644E",  # splashOrange — success / done
        "num": "#FF644E",  # hero numbers
        "link": "#BF3131 underline",  # accentRed — gated link / CTA
        "warn": "#D97706",  # chartAmber
        "err": "bold #BF3131",
        "muted": "#5C4A46",  # inkLight — secondary text
        "status.spinner": "#E76F57",  # override Rich's green default spinner
    }
)

console = Console(theme=BRAND)
err_console = Console(stderr=True, theme=BRAND)

# Brand spinner — a marlin swimming (design ②: mascot-as-motion). Registered
# into Rich's spinner table so console.status(spinner="marlin", …) can use it.
try:
    from rich._spinners import SPINNERS

    SPINNERS.setdefault(
        "marlin",
        {
            "interval": 110,
            "frames": [
                "><>      ",
                " ><>     ",
                "  ><>    ",
                "   ><>   ",
                "    ><>  ",
                "     ><> ",
                "      ><>",
                "     <>< ",
                "    <><  ",
                "   <><   ",
                "  <><    ",
                " <><     ",
                "<><      ",
            ],
        },
    )
except Exception:  # pragma: no cover — spinner is cosmetic; never block on it
    pass

# First-run / --version hero: the gradient block wordmark. Vertical fade from
# splash-orange to accent-red; Rich auto-degrades on non-truecolor terminals
# and honors NO_COLOR. Shown only by setup + version, never on hot-path.
_HERO_LINES = (
    "███╗   ███╗ █████╗ ██████╗ ██╗     ██╗███╗   ██╗",
    "████╗ ████║██╔══██╗██╔══██╗██║     ██║████╗  ██║",
    "██╔████╔██║███████║██████╔╝██║     ██║██╔██╗ ██║",
    "██║╚██╔╝██║██╔══██║██╔══██╗██║     ██║██║╚██╗██║",
    "██║ ╚═╝ ██║██║  ██║██║  ██║███████╗██║██║ ╚████║",
)
_HERO_GRADIENT = ("#FF644E", "#EF5747", "#DF4B40", "#CF3E38", "#BF3131")

_FORCE_JSON = False


def set_json(force: bool) -> None:
    """Set whether command output should be forced to JSON mode."""
    global _FORCE_JSON
    _FORCE_JSON = force


def is_json() -> bool:
    """Return whether stdout should receive JSON."""
    return _FORCE_JSON or not sys.stdout.isatty()


def emit(data: Any, human=None) -> None:
    """Emit machine-readable JSON or human output.

    Parameters
    ----------
    data
        JSON-serializable payload.
    human
        Optional callback for human-mode rendering.
    """
    if is_json():
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
        sys.stdout.flush()
    elif human is not None:
        human()
    else:
        console.print(data)


def status(msg: str) -> None:
    """Write progress to stderr so JSON stdout remains clean."""
    err_console.print(f"[dim]{msg}[/dim]")


def banner() -> None:
    """Print the first-run and version banner."""
    console.print()
    for line, color in zip(_HERO_LINES, _HERO_GRADIENT, strict=True):
        console.print(f"  [{color}]{line}[/]")
    console.print("  [muted]video understanding, on your Mac ·[/muted] [model]Marlin-2B[/model]")
    console.print()


@contextmanager
def spinner(title: str, *, fish: bool = False):
    """Create a stderr progress spinner.

    Parameters
    ----------
    title
        Spinner title.
    fish
        Whether to use the custom marlin spinner.
    """
    if is_json():
        err_console.print(f"[muted]{title}…[/muted]")
        yield lambda m: err_console.print(f"[muted]  {m}[/muted]")
    else:
        name, style = ("marlin", "#FF644E") if fish else ("dots", "model")
        with err_console.status(
            f"[model]{title}…[/model]", spinner=name, spinner_style=style
        ) as st:
            yield lambda m: st.update(f"[model]{title} — {m}…[/model]")


@contextmanager
def build_spinner(title: str):
    """Create an elapsed-time spinner for long build phases.

    Parameters
    ----------
    title
        Spinner title.
    """
    if is_json():
        err_console.print(f"[muted]{title}…[/muted]")
        yield lambda m: err_console.print(f"[muted]  {m}[/muted]")
    else:
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        with Progress(
            SpinnerColumn(spinner_name="dots", style="#E76F57"),
            TextColumn("[model]{task.description}[/model]"),
            TextColumn("[muted]·[/muted]"),
            TimeElapsedColumn(),
            console=err_console,
            transient=True,
        ) as prog:
            task = prog.add_task(title, total=None)
            yield lambda m: prog.update(task, description=f"{title} — {m}")
