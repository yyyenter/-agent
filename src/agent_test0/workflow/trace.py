# agent_test0/workflow/trace.py
"""
Span-based trace: 追踪 Flow 执行中的数据流, 不只是耗时。
"""

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional


class Span:
    """单个调用单元: start/end/duration/attrs/status/children"""

    def __init__(self, name: str, **attrs: Any) -> None:
        self.span_id: str = uuid.uuid4().hex[:8]
        self.parent_id: Optional[str] = None
        self.name: str = name
        self.start: float = time.perf_counter()
        self.end: Optional[float] = None
        self.attrs: dict[str, Any] = dict(attrs)
        self.status: str = "running"
        self.children: list[Span] = []

    def set_output(self, **kwargs: Any) -> "Span":
        for k, v in kwargs.items():
            self.attrs[f"out.{k}"] = v
        return self

    def set_status(self, status: str, **kwargs: Any) -> "Span":
        self.status = status
        for k, v in kwargs.items():
            self.attrs[k] = v
        return self

    def finish(self) -> None:
        if self.end is None:
            self.end = time.perf_counter()

    @property
    def duration(self) -> float:
        if self.end is None:
            return time.perf_counter() - self.start
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "status": self.status,
            "attrs": self.attrs,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    def __init__(self) -> None:
        self.root_spans: list[Span] = []
        self._stack: list[Span] = []
        self._all_spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        s = Span(name, **attrs)
        if self._stack:
            s.parent_id = self._stack[-1].span_id
            self._stack[-1].children.append(s)
        else:
            self.root_spans.append(s)
        self._stack.append(s)
        self._all_spans.append(s)
        try:
            yield s
        except Exception as e:
            s.set_status("error", error=str(e), error_type=type(e).__name__)
            raise
        else:
            if s.status == "running":
                s.set_status("ok")
        finally:
            s.finish()
            if self._stack and self._stack[-1] is s:
                self._stack.pop()

    def tree(self, max_value_len: int = 60) -> str:
        if not self.root_spans:
            return "[Trace] 无 span 记录"
        lines = []
        for r in self.root_spans:
            lines.append(self._format_tree(r, 0, max_value_len))
        total = sum(s.duration for s in self._all_spans)
        all_ok = all(s.status == "ok" for s in self._all_spans)
        lines.append("")
        lines.append(f"[Trace] 总耗时: {total:.2f}s  节点数: {len(self._all_spans)}  "
                     f"状态: {'OK' if all_ok else 'FAIL'}")
        return "\n".join(lines)

    def _format_tree(self, span: Span, depth: int, max_len: int) -> str:
        prefix = "  " * depth
        if depth == 0:
            prefix = ""
        icon = {"ok": "OK", "error": "ERR", "running": "..."}.get(span.status, "?")
        line = f"{prefix}|- [{icon}] {span.name}  {span.duration:.2f}s"
        if span.attrs:
            display_attrs = [(k, v) for k, v in span.attrs.items() if not k.startswith("out.")]
            output_attrs = [(k, v) for k, v in span.attrs.items() if k.startswith("out.")]
            if display_attrs:
                line += f"  | in: {self._fmt_attrs(display_attrs, max_len)}"
            if output_attrs:
                out_kv = [(k[len('out.'):], v) for k, v in output_attrs]
                line += f"  | out: {self._fmt_attrs(out_kv, max_len)}"
        for c in span.children:
            line += "\n" + self._format_tree(c, depth + 1, max_len)
        return line

    @staticmethod
    def _fmt_attrs(kv_list: list, max_len: int) -> str:
        parts = []
        for k, v in kv_list:
            s = _format_value(v, max_len)
            parts.append(f"{k}={s}")
        return ", ".join(parts)

    def dump_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "total_duration": sum(s.duration for s in self._all_spans),
                    "span_count": len(self._all_spans),
                    "root_spans": [s.to_dict() for s in self.root_spans],
                },
                f, ensure_ascii=False, indent=2, default=str,
            )

    def reset(self) -> None:
        self.root_spans.clear()
        self._stack.clear()
        self._all_spans.clear()


def _format_value(v: Any, max_len: int = 60) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if len(v) > max_len:
            return f'"{v[:max_len]}..."'
        return f'"{v}"'
    if isinstance(v, (list, tuple)):
        n = len(v)
        preview = str(v)[:max_len]
        if len(str(v)) > max_len:
            preview = preview + "..."
        return f"[{n} items] {preview}"
    if isinstance(v, dict):
        n = len(v)
        preview = str(v)[:max_len]
        if len(str(v)) > max_len:
            preview = preview + "..."
        return f"{{{n} keys}} {preview}"
    return str(v)[:max_len]


_tls = threading.local()


def _get_tracer() -> Tracer:
    t = getattr(_tls, "tracer", None)
    if t is None:
        t = Tracer()
        _tls.tracer = t
    return t


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    with _get_tracer().span(name, **attrs) as s:
        yield s


@contextmanager
def timed(label: str, **attrs: Any) -> Iterator[Span]:
    with span(label, **attrs) as s:
        yield s


def reset() -> None:
    _get_tracer().reset()


def report() -> None:
    print(_get_tracer().tree())


def dump_json(path: str) -> None:
    _get_tracer().dump_json(path)


def get_all_spans() -> list[Span]:
    return list(_get_tracer()._all_spans)


def quiet_crewai() -> None:
    """
    关闭 CrewAI 自带的 ┌─...└─ 框 + Method Running/Completed UI。

    实现:
      1) crewai.* logger 全部设到 WARNING
      2) monkey-patch ConsoleFormatter.print_panel 为 no-op
         (Flow 的 event_listener 用 verbose=True 实例化, 框就从这里出)
      3) monkey-patch ConsoleFormatter 各种 *_handler 为 no-op
         (Flow Started/Method Running/Completed 事件处理)
      4) 设 CREWAI_TRACING_ENABLED=false
    """
    import logging
    for name in ("crewai", "crewai.flow", "crewai.events", "crewai.agents",
                 "crewai.tasks", "crewai.utilities"):
        logging.getLogger(name).setLevel(logging.WARNING)
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    # 静默 rich Panel 框 (Flow/Method Running/Completed)
    try:
        from crewai.events.utils.console_formatter import ConsoleFormatter
        # 直接 no-op 全部 print_panel 调用
        ConsoleFormatter.print_panel = lambda self, *a, **kw: None
        # 静默各种事件 handler
        for name in (
            "update_span_status", "create_panel", "create_flow_panel",
            "handle_flow_started_event", "handle_flow_finished_event",
            "handle_method_execution_started_event",
            "handle_method_execution_finished_event",
            "handle_method_execution_failed_event",
        ):
            if hasattr(ConsoleFormatter, name):
                setattr(ConsoleFormatter, name, lambda self, *a, **kw: None)
    except ImportError:
        pass


__all__ = [
    "Span", "Tracer",
    "span", "timed", "reset", "report", "dump_json", "get_all_spans",
    "quiet_crewai",
]
