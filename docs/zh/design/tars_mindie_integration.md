# TARS 与 MindIE Motor 集成设计

## 分层与所有权

TARS 集成分为纯算法、Conductor 证据适配、有状态运行时、可靠性交付协议和 Motor 激活桥五层。
纯算法与协议层不直接访问 HTTP、ZMQ 或 InstanceManager；激活桥只在
`prefill_scheduler_type=tars` 且 `decode_scheduler_type=tars` 时创建，并且是内部预留与
Motor workload 的唯一映射点。配置加载和运行时入口均拒绝单侧启用；旧 `scheduler_type`
只作为未显式配置角色的默认值。

Scheduler 进程是 admission、最终目标、容量和协议状态的权威所有者。Router 只保存已交付但
未终态的轻量记录，负责确认、撤销和生命周期事件。Motor 原有 RequestManager 继续拥有实际
P/D workload 的请求级释放记录，避免建立第二套外部负载账本。

## 身份与证据

每次尝试使用 `(request_id, attempt_seq, scheduler_epoch)`；每个目标使用
`(Motor instance_id, endpoint.id)`。与 Conductor 对账时，Motor instance 先转换为当前
Conductor instance identity，`endpoint.id` 原样作为 `dp_rank`。只有 workers 注册快照和查询
响应对预期的 model、tenant、block size、instance 和 dp_rank 全部闭合，证据才可用。

这不是节点级 locality。Pod IP 或 backend IP 不能可靠表示 Kubernetes `spec.nodeName`；因此当前
实现不会把“相同 IP”扩展解释为“相同物理节点”。

## 可靠性交付

单角色与 Bootstrap 联合分配都采用 prepare、Router persist、confirm 三阶段交付。prepare
期间的 workload 对 Router 不可见；未确认交付受 lease 限制，到期由 Scheduler 撤销并补偿。
confirm/revoke 响应不确定时使用 status 对账。联合分配以一个 token 同时覆盖 P/D，禁止拆成
两个单角色请求。

正常生命周期为 `prefill_complete -> decode commit（仅延迟 Decode 模式） -> request_complete`；
异常生命周期为 `abort`。内部 TARS 状态终结后，Router 再释放 Motor 的实际 workload。

## 兼容和升级

非 TARS 策略继续使用原有 `ALLOCATE_ONLY`，未注册任何 TARS RPC handler。TARS wire version
当前没有跨版本兼容层，因此升级必须保证一个 Coordinator Pod 内的 Scheduler 与全部 Router
worker 来自同一 wheel/image，并整体替换。初始版本不启用 Coordinator 主备，不声称进程崩溃后
恢复 in-flight TARS 状态。
