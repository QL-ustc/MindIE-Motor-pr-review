# [2026-08-29] Bootstrap 分腿分配可能暴露不完整 P/D 所有权

- **现象 (Symptom)**：若把 Handoff 的单角色确认协议调用两次，或沿用生产 Router 现有的 P 后 D 顺序独立落账，Bootstrap 在 P 成功而 D 失败、Router 持久化中断或确认丢失时，可能留下单角色工作负载，或者让 Router 看见不完整的 P/D 对。
- **根因 (Root cause)**：Bootstrap 要求 P/D upfront，但单角色协议只有一个 target 和一份 workload；它无法表达“两份本地账本均已提交后，用一个 token 转移联合所有权”的原子可见性边界，也无法按 D 后 P 的顺序补偿。
- **为什么会写出 (Why)**：把“复用可靠单角色协议”误当成“组合后自然获得联合原子性”，忽略两个成功响应之间仍存在可独立失败的持久化、容量和确认边界；同时忽略 Trigger、Handoff 与 Bootstrap 的资源时序不同。
- **修复 (Fix)**：新增休眠的 `bootstrap_joint_protocol.py`。无副作用 preflight 先限制 Bootstrap 模式和有效 epoch，再分配新 attempt；P 经 TARS 选择，D 经注入的纯选择回调保持 master 兼容；两边最终校验后依次落账，只返回一个 pending P/D pair。Router 以一个 delivery token 确认后才能成对可见，确认前故障按 D 后 P 反向撤销，确认后生命周期释放剩余工作负载。
- **测试拦截 (Test interception)**：`test_tars_bootstrap_joint_protocol.py` 验证确认前不可派发、P/D 成对可见、非 Bootstrap 不消耗 attempt、重复 admission 不 join/takeover、类型化目标校验；运行 10,000 次状态模拟、1,000 次并发请求，并对 9 个边界各注入 100 次故障，每次终止均断言 attempt、pending delivery、P reservation/workload、D workload 和 Router ledger 归零。
- **场景 (Scenario)**：SGLang Bootstrap upfront P/D 分配、D 容量或熔断变化、Router 持久化/确认失败、Prefill 结束与请求终止、并发请求和重试 attempt。
- **关键词 (Keywords)**：TARS、Bootstrap、P/D atomic visibility、joint delivery、reverse compensation
