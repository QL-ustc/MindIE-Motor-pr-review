# [2026-08-29] Handoff 分配缺少可证明的确认、超时与重启隔离边界

- **现象 (Symptom)**：旧实验实现把 Scheduler reservation、Router 投递和 P/D 分配散落在异步路径中；迁移时无法证明 Prefill 在 Router 持久化目标前不会被派发，也无法证明 Prefill 完成前 Decode 不会被提前分配。调用超时、取消、交付确认丢失或 Scheduler 非优雅重启后，还可能遗留 reservation、workload 或 Router 账目并错误恢复接流。
- **根因 (Root cause)**：协议没有携带 `(request_id, attempt_seq, scheduler_epoch)` 的唯一 attempt 身份，也没有把 pending delivery、ownership transfer、Prefill settlement 和 Decode commit 建模为带类型化结果的显式状态转换；同时缺少严格的 client/server/stage 超时序、交付租约，以及 `RUNNING/DRAINING/STOPPED/RECOVERING` 生命周期。
- **为什么会写出 (Why)**：把一次函数返回视为跨进程所有权已转移，忽略 Scheduler 成功、Router 持久化、确认响应及实际派发之间均可能独立失败；把 RPC 超时误当成业务已取消；把进程重启误当成可以恢复内存事务；同时把 P/D 统一预分配错误套用到 P-first、D-deferred 的 Handoff 时序。
- **修复 (Fix)**：dormant `handoff_protocol.py` 由 Scheduler 协议服务持有 attempt、pending-delivery、Decode workload 和 terminal receipt，Router 账本确认后才能派发；Prefill 幂等释放后才允许 Decode commit，并在提交前重查 topology、circuit、role 和 engine type。新增严格超时预算、交付租约/状态对账、原始取消传播、排空截止时间和 Scheduler epoch 冷隔离；非优雅重启只有在 cooldown、Router 账本清空和操作员显式确认后才能恢复接流，不宣称事务恢复。
- **测试拦截 (Test interception)**：`test_tars_handoff_protocol.py` 覆盖确认前派发拒绝、提前 Decode 拒绝、重复请求/确认/撤销/生命周期幂等和类型化失败；对 8 个协议边界各注入 100 次失败并验证瞬态账本归零，连续执行 1000 个 attempt 验证三方身份一致，并执行 10,000 次乱序超时/取消竞争、50 个非优雅重启切点、交付租约过期和排空截止测试，检查零重复派发及零瞬态泄漏。
- **场景 (Scenario)**：Handoff P-first/D-deferred、Router 持久化或确认中断、调用方取消、交付租约到期、排空、Prefill 回调重试、Decode 目标熔断/拓扑更新/引擎不匹配、Scheduler epoch 切换和非优雅重启。
- **关键词 (Keywords)**：TARS、Handoff、delivery lease、scheduler epoch、cancellation compensation
