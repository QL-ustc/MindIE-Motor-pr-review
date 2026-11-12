# [2026-08-28] TARS 旧身份和介质复用语义会制造错误 DP 缓存信用

- **现象 (Symptom)**：旧 TARS 适配以候选下标和 endpoint 参与调度映射，并把 XPU/CPU/Disk 或 pool fanout 结果解释为 local/remote；请求身份还会读取多个 legacy body 字段，容易在 endpoint 变化、DP>1 或非精确事件下产生身份漂移和错误缓存信用。
- **根因 (Root cause)**：旧实现将传输映射、存储介质和逻辑调度身份混在同一模型中，并依赖候选数量的外部 hash ring 和全局随机 tie-break；Conductor response 的字段形状又被用来推断 DP 所有权，而不是消费服务端 provenance。
- **为什么会写出 (Why)**：把“某介质报告命中”误当成“命中属于这个 DP”，把 HTTP 请求体长度误认为可替代 token count，并低估了 candidate order、endpoint/IP 和进程 hash 对冷启动稳定性的影响。
- **修复 (Fix)**：新增 dormant 纯 TARS 数据路径：唯一 target 是 `(str(instance.id), dp_rank)`；session 只取 `agent_hint_info.session_id`，缺失时显式退回 request ID；token count 只取 `len(token_ids)`；双哈希使用无 RNG 状态的稳定 BLAKE2 target scoring；Conductor adapter 只为 `exact_dp_npu_provenance=true` 的 NPU blocks 计算 best-effort credit，CPU/Disk 仅保留诊断。
- **测试拦截 (Test interception)**：算法 scorer/selector 金标覆盖 cache-first 和 overload-to-min-TTFT；10,000 session 检查 target identity 分布和稳定性，10,000 generated evidence 检查零 false exact-DP attribution 及 `compute + credit = input`；测试还覆盖缺 token IDs、request-ID fallback、sparse zero/unavailable、CPU/Disk 零信用和 import/construct 零网络/线程副作用。
- **场景 (Scenario)**：DP>1、endpoint/IP 变化、同 instance 多 DP、IpOnly pool 事件、Conductor timeout/malformed response、agent 请求缺 session ID，或后续把纯 TARS core 接入运行时。
- **关键词 (Keywords)**：TARS、stable dual hash、instance_id dp_rank、exact DP provenance、cache credit
