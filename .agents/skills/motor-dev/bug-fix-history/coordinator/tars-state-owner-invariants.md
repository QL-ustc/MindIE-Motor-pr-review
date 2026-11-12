# [2026-08-28] 分散 TARS 状态无法证明请求唯一归属与最终目标稳定

- **现象 (Symptom)**：旧实验实现把队列、reservation、workload 和 replica pending/running 分散在多个容器及异步路径中；迁移到新 Scheduler 时，无法通过单一边界证明一次 attempt 只属于一个目标，也无法证明拓扑变化后已锁定目标不会被陈旧重平衡改写。
- **根因 (Root cause)**：旧实现缺少以 `(request_id, attempt_seq)` 为键、由 Scheduler 唯一持有的状态聚合；拓扑版本、队列归属、容量扣减和最终目标锁定不是同一原子状态转换的一部分。
- **为什么会写出 (Why)**：把各阶段局部字典视为等价于统一状态机，忽略了拓扑回调、重平衡和完成回调交错时必须同时维护的跨索引不变量。
- **修复 (Fix)**：新增 dormant `TarsStateRuntime`，集中持有 attempt、queue、reservation、workload 和 replica 索引；只接受递增 topology epoch，最终目标锁定前必须先通过容量检查且一经锁定不可重选，reservation 原子完成，并提供统一 completion/abort/quiesce 清理和内部不变量检查。
- **测试拦截 (Test interception)**：`tests/coordinator/scheduler/test_tars_state_runtime.py` 覆盖唯一 Scheduler owner、fake-clock pending TTFT、容量逃逸与防环、陈旧/重复拓扑拒绝、final-target 锁定及清理；固定种子 10,000 次随机转换逐步检查唯一队列/归属和 final target 不变，静默后瞬态状态为零而活跃拓扑保留。
- **场景 (Scenario)**：DP>1 目标、容量耗尽触发重平衡、拓扑滚动更新、prefill 到 running 的生命周期转换、重复完成/中止，或后续接入 ACK/ABORT 协议。
- **关键词 (Keywords)**：TARS、state owner、topology epoch、final target lock、quiescence
