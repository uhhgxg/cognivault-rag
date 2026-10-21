"""用于跨流水线阶段可观测性的追踪上下文。

提供 trace_id、trace_type（query/ingestion）、每个阶段的计时、
finish() 生命周期，以及用于 JSON Lines 输出的 to_dict() 序列化。
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TraceContext:
    """请求级别的追踪上下文，记录流水线各阶段及其耗时。

    属性：
        trace_id: 本次追踪的唯一标识。
        trace_type: 取值为 ``"query"`` 或 ``"ingestion"``。
        started_at: 追踪创建时的 ISO-8601 时间戳。
        finished_at: 调用 ``finish()`` 时的 ISO-8601 时间戳，未调用则为 None。
        stages: 已记录的阶段字典的有序列表。
        metadata: 附加到本次追踪的任意键值对。
    """

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = field(default=None)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 内部单调时钟，用于精确计算耗时
    _start_mono: float = field(default_factory=time.monotonic, repr=False)
    _finish_mono: Optional[float] = field(default=None, repr=False)
    _stage_timings: Dict[str, float] = field(default_factory=dict, repr=False)

    # ---- 记录 -------------------------------------------------------

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """记录某个流水线阶段的数据。

        参数：
            stage_name: 阶段名称（例如 ``"dense_retrieval"``）。
            data: 阶段专属的数据负载（方法、provider、详情等）。
            elapsed_ms: 预先计算好的耗时（毫秒）。如果为 *None*，
                调用方应自行外部计时，或交给 ``stage_timer`` 上下文管理器处理。
        """
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    # ---- 生命周期 ---------------------------------------------------

    def finish(self) -> None:
        """标记追踪为已结束，并记录结束的真实时间。"""
        self._finish_mono = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    # ---- 计时辅助 ---------------------------------------------------

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """返回耗时（毫秒）。

        参数：
            stage_name: 如果提供，则返回该阶段已记录的耗时。
                如果为 *None*，则返回整条追踪的总耗时
               （开始 → 结束；若尚未结束则为 开始 → 当前）。

        返回：
            耗时（毫秒）。

        抛出：
            KeyError: 提供了 *stage_name* 但未找到对应记录时抛出。
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"Stage '{stage_name}' has no recorded timing")
            return self._stage_timings[stage_name]

        end = self._finish_mono if self._finish_mono is not None else time.monotonic()
        return (end - self._start_mono) * 1000.0

    # ---- 序列化 -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """将追踪序列化为普通字典，可直接用于 ``json.dumps``。

        返回：
            包含全部追踪数据的字典。
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round(self.elapsed_ms(), 2),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }

    # ---- 向后兼容辅助（用于 C5 / C6）--------------------------------

    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """获取某个阶段已记录的数据。

        在 stages 列表中查找（同名阶段以最后写入为准）。

        参数：
            stage_name: 要获取的阶段名称。

        返回：
            匹配阶段的 ``data`` 字典；未找到则返回 *None*。
        """
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None
