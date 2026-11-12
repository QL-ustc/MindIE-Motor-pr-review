# TARS 调度

TARS 是面向 KV 前缀复用和排队成本的有状态调度策略。Motor 在 Scheduler 进程中维护唯一的
TARS 状态，以 `(instance_id, endpoint.id)` 作为调度目标；其中 `endpoint.id` 必须等于引擎
注册到 KV Conductor 的 `dp_rank`。只有精确且完整的 DP 级 NPU 证据可抵扣 Prefill 计算量，
CPU、Disk 和无法归属到精确 DP 的命中只用于诊断，不参与 TARS cache credit。

## 启用配置

```json
{
  "motor_coordinator_config": {
    "standby_config": {"enable_master_standby": false},
    "scheduler_config": {
      "prefill_scheduler_type": "tars",
      "decode_scheduler_type": "tars",
      "tars": {
        "fallback_policy": "load_balance",
        "enable_decision_log": false
      }
    }
  },
  "kv_conductor_config": {
    "conductor_service": "kv-conductor",
    "http_server_port": 13333,
    "model_path": "qwen3-8B",
    "block_size": 128
  }
}
```

`prefill_scheduler_type` 和 `decode_scheduler_type` 必须同时为 `tars`，包括 PDHybrid 场景。
只启用一侧会在配置校验阶段被拒绝。旧配置的 `scheduler_type: tars` 仍会填充未显式设置的
P/D 字段；显式角色字段优先，迁移后的配置也必须满足两侧同时启用的约束。

两侧均不为 `tars` 时，不创建 TARS 队列、状态、后台任务或 RPC，也不会额外查询
KV Conductor。所有 TARS 配置字段都是启动期配置；修改后必须按原子升级方式同时更新
Coordinator 内的 Scheduler 和 Router worker，不支持新旧 TARS wire protocol 混跑。

完整示例：

- `examples/features/tars_p4d4`：4 个 Prefill 实例和 4 个 Decode 实例，每实例 DP1。
- `examples/features/tars_kv_pool_dp2`：2P×DP2 和 2D×DP2，使用当前顶层
  `kv_cache_store_config` 与 `kv_conductor_config`。

## 调度时序

| 模式 | 分配顺序 | TARS 事务边界 |
|---|---|---|
| vLLM Trigger | Decode 先分配；metaserver 回调时再分配 Prefill | 仅 Prefill 使用单角色确认；Decode 保持现有 Trigger 路径 |
| vLLM Handoff | Prefill 先分配；Prefill 完成后再分配 Decode | Prefill 使用单角色确认；Decode 延迟分配 |
| SGLang Bootstrap | Prefill 和 Decode 必须在请求派发前同时可见 | 使用一个 P/D 联合确认；任一角色失败都会逆序补偿 |
| PDHybrid | 一个 Union 实例处理请求 | Union 使用单角色确认 |

Scheduler 先提交内部预留和 Motor workload，再向 Router 返回待确认交付；Router 本地持久化
交付记录并确认后，目标才可派发。确认响应丢失时 Router 必须查询 Scheduler 状态，不能直接
撤销或重复提交。正常结束依次执行 Prefill 结算、必要的 Decode 绑定、请求终态和 Motor workload
释放；取消或异常使用 abort 终态。

## 失败与降级

只有在任何 TARS 状态或 workload 尚未修改、且 KV Conductor 精确证据不可用时，才允许使用
`fallback_policy`。一旦 admission 已开始，超时、身份不一致、确认失败和联合分配失败都按
fail-closed 处理，不再静默降级。

初始版本的部署边界为：一个 KV Conductor 副本、一个 active Coordinator、单模型、固定 tenant
和统一 block size。Coordinator 停止时会停止接收 admission、等待当前处理任务并清理进程内状态；
不声明 SIGKILL、Pod 丢失或跨版本非优雅恢复能力。发生非优雅重启后必须先隔离旧 Router，清空
旧连接，再整体重建 Coordinator。

## 当前不支持

- “同一 Kubernetes 物理节点”为 local 的节点级 locality。当前 Conductor 身份只能证明
  backend/DP 归属，不能证明 Pod 所在物理节点。
- 将同节点其他 DP 的命中合并为 local。当前 cache credit 严格限定为候选 DP 的精确 NPU 命中。
- 多 active Coordinator、动态 tenant、多模型共享一个 TARS 状态，以及非优雅状态恢复。

节点级 locality 需要独立的、可验证的 Pod-to-node 身份协议和影子验证，不能从 Pod IP 推断。

## D1 实机证据门禁

启用 `enable_decision_log` 后，Coordinator 以单行紧凑 JSON 记录 `prepared`、`confirmed`、
`revoked`、`lifecycle` 和 pre-admission `fallback`，记录包含 `req_id`、`attempt_seq`、
`scheduler_epoch` 及精确 `(instance_id, dp_rank)` 目标，不包含 prompt 或 token。

实机运行后可使用 `scripts/validate_tars_d1_evidence.py` 校验候选 SHA、wheel/image/config/trace
指纹、请求结果、决策生命周期，以及 `/workers` 中预期 DP 的唯一注册和
`exact_dp_npu_provenance`：

`candidate.json` 中所有 artifact 路径均相对于该文件所在目录：

```json
{
  "final_candidate_sha": "<40位Git SHA>",
  "image_digest": "sha256:<64位摘要>",
  "versions": {
    "motor": "3.1.0",
    "vllm": "<版本>",
    "cann": "<版本>",
    "driver": "<版本>",
    "os": "<版本>",
    "hardware": "<型号>",
    "model": "<模型>",
    "backend": "<KV backend及版本>"
  },
  "thresholds": {
    "conductor_query_p99_ms": 20,
    "non_tars_success_rate_drop_pp": 0.1,
    "cold_request_ttft_p95_regression_pct": 5
  },
  "wheel_artifacts": [{"path": "motor.whl", "sha256": "<64位摘要>"}],
  "config_artifacts": [{"path": "user_config.json", "sha256": "<64位摘要>"}],
  "trace_artifacts": [{"path": "results.jsonl", "sha256": "<64位摘要>"}]
}
```

`expected-workers.json` 是对象数组，每项声明一个预期 P/U DP；精确 DP/NPU 模式必须将
`require_exact_dp_npu_provenance` 设为 `true`。`results.jsonl` 每行至少包含唯一的 `req_id`
和 `status_code`。

```json
[
  {
    "instance_id": "vllm-prefill-0",
    "dp_rank": 0,
    "require_exact_dp_npu_provenance": true
  }
]
```

```bash
python scripts/validate_tars_d1_evidence.py \
  --candidate-manifest artifacts/candidate.json \
  --repo . \
  --decision-log artifacts/coordinator.log \
  --request-results artifacts/results.jsonl \
  --workers-snapshot artifacts/workers.json \
  --expected-workers artifacts/expected-workers.json \
  --expected-requests 1000 \
  --topology p4d4 \
  --mode handoff \
  --report artifacts/d1-report.json
```

该门禁只覆盖证据包完整性、请求与决策身份闭合、生命周期终态和 Worker 身份。它不能证明
真实 NPU 数据面、Router 实际派发目标、Scheduler 内部静默状态、故障/重启、性能/浸泡或真实
回滚；这些项目仍必须按 P4D4 与 KV Pool DP>1 的 D1 实机流程分别验收，脚本 `PASS` 不等于
`CFF_MIGRATION_ACCEPTED`。冻结候选指纹前仓库必须为 clean；脚本会拒绝将脏工作树产物绑定到
旧的 `HEAD`。
