# [2026-08-28] vLLM 0.23 固定位置事件中的 nil 导致字段错位

- **现象 (Symptom)**：vLLM 0.23 发送合法 `BlockStored` 数组时，`group_idx` 会被读成 `lora_id`，`kv_cache_spec_kind` 会被读成 `medium`；`FullAttention` HBM 事件因此可能被当成非 HBM 事件而不进入查询树，SWA/Mamba 事件也可能绕过过滤。畸形 `BlockStored` 还可能被部分接受。
- **根因 (Root cause)**：`motor/kv_conductor/src/events/vllm.rs::parse_block_stored_values` 在解析前统一删除了所有 `nil`。vLLM 0.23 的 msgspec `array_like=True` 编码会为非默认但可空字段保留显式 `nil`，删除它们会破坏后续字段的固定位置；`BlockRemoved` 的 `medium=nil` 也有同类问题。
- **为什么会写出 (Why)**：错误地把 `omit_defaults=True` 理解为所有空字段都会从数组中消失。实际上它只允许省略尾部默认值，必需的可空字段仍占据固定位置；类型模式解析不能在删除占位符后继续保证字段语义。
- **修复 (Fix)**：先识别 vLLM 0.23 固定位置布局，按精确下标读取 `BlockStored` 和 `BlockRemoved` 字段，并校验必需字段、可空字段和尾部字段类型；识别失败时才回退到原有紧凑布局，保留旧发布端兼容性。未知事件仍按忽略处理。
- **测试拦截 (Test interception)**：新增固定位置 `BlockStored`/`BlockRemoved` wire golden，覆盖 HBM 查询命中、FullAttention/SWA/Mamba 过滤、`medium=nil` 后的 `group_idx`、畸形必需字段与非法可空字段；原有紧凑布局用例继续通过。`cargo test` 验证 128 个单元测试和 20 个集成测试全部通过。
- **场景 (Scenario)**：vLLM 0.23 使用 msgspec `array_like=True`、tagged union、`omit_defaults=True` 发布 KV 事件，且 `parent_block_hash`、`lora_id`、`medium` 或 `lora_name` 为 `nil`。
- **关键词 (Keywords)**：kv_conductor、vLLM 0.23、msgspec、nil placeholder、BlockStored
