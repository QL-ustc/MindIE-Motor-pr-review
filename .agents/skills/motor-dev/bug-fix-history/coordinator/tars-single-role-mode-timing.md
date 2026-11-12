# [2026-08-28] 统一分配流程破坏 Trigger 与 PDHybrid 的真实时序

- **现象 (Symptom)**：把同一套 P/D 预分配流程迁入所有模式会让 Trigger 在 Decode 回调前提前分配 Prefill；旧 ROLE_U 路径还会用首个响应或近似时间冒充物理 `prefill_complete`，并可能让旧重试的生命周期消息作用于新 attempt。
- **根因 (Root cause)**：协议层没有显式区分 Trigger 的 D-first/P-callback 时序与 PDHybrid 的单个 `UNIFIED` 角色，也没有把 `(request_id, attempt_seq, scheduler_epoch)` 和实际存在的生命周期边界绑定。
- **为什么会写出 (Why)**：把“统一接口”误解为“统一执行时序”，忽略引擎模式决定资源何时可见；同时把逻辑上的请求进度错当成了可观测的物理 Prefill 完成证据。
- **修复 (Fix)**：新增 dormant `single_role_adapters.py`。Trigger 先保留空身份且不修改 TARS 状态，仅在串行 metaserver 回调内确认 P 后派发，并校验 P/D 引擎和拓扑；PDHybrid 仅接收 `UNIFIED` admission，每次重试使用新序号，只发送 terminal lifecycle。底层协议增加 `UNIFIED` 角色和对应 terminal 清理规则。
- **测试拦截 (Test interception)**：`test_tars_single_role_adapters.py` 分别执行 1000 次 Trigger 和 PDHybrid attempt，覆盖无提前 P admission、重复及 64 路并发回调、取消/晚到/引擎和拓扑不匹配、重试身份隔离、错误 epoch，以及 ROLE_U 永不产生物理 `prefill_complete`；每次终止后断言所有瞬态账本归零。测试注入固定单调时钟，避免两次无副作用快照比较被实时递减的 `pending_ttft_remaining` 制造并行抖动。
- **场景 (Scenario)**：vLLM layerwise Trigger metaserver 重试或断连、D 已终止后的晚到 P 回调、PDHybrid 节点故障重试、旧 attempt 生命周期消息延迟到达。
- **关键词 (Keywords)**：TARS、Trigger、PDHybrid、D-first、prefill_complete
