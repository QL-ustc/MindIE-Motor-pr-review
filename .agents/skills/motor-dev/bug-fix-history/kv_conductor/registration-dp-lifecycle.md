# [2026-08-28] KV-Conductor 重复注册产生 DP 别名和跨 DP 缓存清理

- **现象 (Symptom)**：同一 `(instance_id, dp_rank)` 重复注册会在 HBM IP 索引中累积重复项；同 IP 迁移后留下旧映射；同一 instance 的不同 DP 使用不同 backend 时，重注册一个 DP 可能清理另一个 DP 的 CPU/Disk 状态；等待旧 ZMQ subscriber 退出时还会阻塞无关 worker 的注册。
- **根因 (Root cause)**：`motor/kv_conductor/src/registry.rs` 将 endpoint 按 DP 保存，却只在 instance 层保存一个会被最后一次注册覆盖的 `store_backend`；IP 索引只追加、不去重，同 backend 重注册又保留无法证明属于当前进程代际的 HBM；注册和注销还在持有全局 subscriber-map 写锁时等待 `shutdown()`。
- **为什么会写出 (Why)**：把可变 endpoint 和 backend 当作 instance 身份，把“地址未变”误当成“进程代际未变”，并忽略了 DP>1 下共享 instance 元数据与精确 DP 缓存所有权是两个不同层次。
- **修复 (Fix)**：缓存、endpoint、backend 和 subscriber 清理均以 `(instance_id, dp_rank)` 为目标；IP 索引幂等去重并清除旧地址；每次重注册保守清除该 DP 的 NPU 且不 replay，backend 变化时清除该 DP 的全部介质；model/tenant/block-size 冲突返回 typed error 和 HTTP 409；subscriber 先从全局表摘除，再在锁外等待退出。共享 namespace 由 per-instance 生命周期锁串行保护。
- **测试拦截 (Test interception)**：注册表单测覆盖 1,000 次相同重注册、10,000 次确定性随机注册/注销、精确 DP NPU 清理、跨 DP backend 隔离、失败不变性、无孤儿锁、live/replay matcher 一致性和并发 shutdown 屏障、replay 不持有 global `instances` 写锁；HTTP 集成测试覆盖 identity conflict 409 与同/异 backend 的旧 HBM 清理。`cargo clippy --all-targets -- -D warnings` 与注册相关单测/集成测试通过；测试数量随套件演进，不在案例中固化。
- **场景 (Scenario)**：DP>1 worker 重启、endpoint/IP 迁移、同 instance 各 DP backend 不同、控制面重复注册，或旧 subscriber 退出延迟。
- **关键词 (Keywords)**：kv_conductor、exact DP、idempotent registration、stale IP index、subscriber lock
