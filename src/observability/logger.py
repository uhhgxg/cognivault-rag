"""可观测性日志工具。

提供：
- ``get_logger``：标准的人类可读日志器（沿用 C 阶段实现）。
- ``JSONFormatter``：自定义的 :class:`logging.Formatter`，输出 JSON 格式。
- ``get_trace_logger``：返回一个由 JSON Lines 文件处理器支持的日志器。
- ``write_trace``：便捷函数，将一个 trace 字典追加到
  ``logs/traces.jsonl``。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import resolve_path

# 默认的 traces 文件路径（绝对路径，不依赖当前工作目录）
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


# ── 人类可读日志器（已有实现） ────────────────────────────────


def get_logger(name: str = "modular-rag", log_level: Optional[str] = None) -> logging.Logger:
    """获取一个配置好的日志器。

    Args:
        name: 日志器名称。
        log_level: 可选的日志级别字符串（例如 "INFO"）。

    Returns:
        配置好的日志器实例。
    """

    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # 屏蔽 httpx 的日志（其中包含敏感的接口地址）
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger(name)


# ── JSON Lines 格式化器 ────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """每行输出一个 JSON 对象的日志格式化器。

    每条日志记录会被序列化为一个字典，至少包含：
    ``timestamp``、``level``、``logger``、``message``。如果记录中
    携带了 ``exc_info`` 元组，则把异常堆栈信息放入
    ``exception`` 字段。

    通过日志调用时 *extra=* 附加的额外属性会被合并到顶层字典中
    （Python 内部字段除外）。
    """

    _INTERNAL_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """把日志记录作为单行 JSON 字符串返回。"""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 合并调用方附加的额外字段
        for key, val in record.__dict__.items():
            if key not in self._INTERNAL_ATTRS and key not in payload:
                try:
                    json.dumps(val)  # 低成本的序列化能力测试
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ── Trace 日志器 ────────────────────────────────────────────────────


def get_trace_logger(
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
    *,
    name: str = "modular-rag.trace",
) -> logging.Logger:
    """返回一个把 JSON Lines 写入 *traces_path* 的日志器。

    该日志器使用 :class:`JSONFormatter` 和一个配置为追加模式的
    :class:`FileHandler`。用相同的 *name* 多次调用会返回同一个
    日志器（标准的 :mod:`logging` 行为）。

    Args:
        traces_path: JSONL 输出的文件路径。父目录会被自动创建。
        name: 日志器名称。

    Returns:
        一个可直接输出 JSON Lines 的 :class:`logging.Logger`。
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免在重复调用时添加重复的处理器
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # 不输出到控制台

    return logger


# ── trace 字典的便捷写入器 ──────────────────────────────


def write_trace(
    trace_dict: Dict[str, Any],
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
) -> None:
    """把一个 trace 字典作为一行 JSON 追加写入。

    这是一个直接写入文件的薄封装——不涉及日志框架——
    因此输出内容与 :class:`~src.core.trace.trace_collector.TraceCollector`
    产生的完全一致。

    Args:
        trace_dict: 一个可被 JSON 序列化的字典（通常来自
            ``TraceContext.to_dict()``）。
        traces_path: 输出文件路径；父目录会被自动创建。
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(trace_dict, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")