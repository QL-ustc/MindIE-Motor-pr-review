# [2026-08-28] Conductor 将不兼容或非精确事件误当成可调度的 DP 证据

- **现象 (Symptom)**：pool 注册使用静态 `model_path`、DP 注册和 query 使用运行时 model，造成同一拓扑落入不同 namespace；`/workers` 为空或失败时会回退到旧 `/services` schema，Coordinator 无法区分冷注册表、协议错误和控制面不可用；进程内 `_pool_registered` 在 Conductor 重启后仍为 true，pool 不再注册；`IpOnly` 节点 IP 扇出写入 NPU 后，query 又没有字段表明该命中不能作为精确 DP 信用。
- **根因 (Root cause)**：`motor/coordinator/api_client/conductor_api_client.py` 没有统一 register/query/unregister/re-register 的 model/tenant/block-size 契约，并将 transport/schema failure 归一化成空列表；`motor/kv_conductor/src/registry.rs` 的 query response 只表达命中数量，不表达事件入口是否能证明 exact DP；pool/vLLM 事件应用也未隔离 `IpOnly` 的 NPU 写入。
- **为什么会写出 (Why)**：把服务存活、注册存在、缓存就绪和 placement provenance 四种不同证据合并成一个“有/无结果”布尔判断，又从 `DP` map 形状反推所有权，忽略了 node-IP fanout 与 exact-DP subscriber 的信任等级不同。
- **修复 (Fix)**：pool 与 DP 统一使用单一运行时 model，混合 model fail closed；reconciliation 只解析当前 `/workers` lowercase schema，空注册表触发 pool 和全部缺失 DP 的冷重建，错误/冲突不做写入；Rust 为 endpoint/query tuple 增加服务端 `exact_dp_npu_provenance`，direct subscriber 可为 true，HTTP NPU 会降级为 false，`IpOnly` stored event 不得写 NPU；Python 新增独立严格 evidence adapter，区分 sparse zero 与 unavailable，并固化 admission 前零 TARS mutation 才能走 LoadBalance 的 fallback 边界。
- **测试拦截 (Test interception)**：Python 测试覆盖 schema/namespace 混用、缺注册、query failure、provenance 缺失/false、sparse zero、fallback 边界、运行时 model 一致性、Conductor restart 冷重建和 KVA 不回归；Rust 测试覆盖 direct=true、pool=false、HTTP taint、JSON/msgpack 字段、sparse zero 与 `IpOnly` NPU 拒绝。`cargo test` 为 141 个单测和 21 个 HTTP 集成测试全通过，针对性 Python 回归 169 项全通过。
- **场景 (Scenario)**：Coordinator 与 Conductor 独立重启、Mooncake/Memcache pool 模式、DP>1、滚动升级期间 schema 不一致、HTTP 与 ZMQ 事件源混用，或 TARS 在 admission 前读取 KV 证据。
- **关键词 (Keywords)**：kv_conductor、exact_dp_npu_provenance、IpOnly、sparse zero、fallback boundary

## Query coverage 字段契约

- **现象 (Symptom)**：Conductor 正常返回的 JSON/MessagePack 命中包含 `npu_end`、`cpu_end`、`disk_end`，Python adapter 却判为 `invalid_query_schema`，使精确 DP 命中整体不可用。
- **根因 (Root cause)**：`conductor_evidence.py` 的 DP 白名单缺少 Rust `DpBlocks` 和 MessagePack encoder 已输出的覆盖终点字段；手写的 Python 测试响应也省略这些字段，掩盖了跨语言契约不一致。
- **为什么会写出 (Why)**：客户端测试只复述客户端假设，没有用服务端序列化契约校验共享样例。
- **修复 (Fix)**：覆盖终点作为已知可选元数据处理，存在时必须是非负整数且不能是布尔值。四个原计数字段仍必填；credit 只来自 exclusive block counts 与注册、查询两份 provenance，不使用覆盖终点重复计数。
- **测试拦截 (Test interception)**：Python 解码共享 JSON/MessagePack 样例并验证精确 DP 信用，同时覆盖错误元数据、缺少必填计数和不含覆盖终点的兼容响应；Rust serializer 测试使用同一组样例约束 wire schema 演进。
- **场景 (Scenario)**：Coordinator 消费带有绝对覆盖终点的 Conductor 查询响应，或双方独立演进 query schema。
- **关键词 (Keywords)**：query schema、coverage ends、shared fixture、MessagePack、exact DP
