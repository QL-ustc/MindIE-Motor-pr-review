# [2026-08-31] 旧产品名门禁把自身规则和测试样例判为迁移违规

- **现象 (Symptom)**：执行 `python scripts/check_tars_migration_paths.py --base upstream/master` 时，门禁报告 `LEGACY_PATTERN`、脚本说明和测试 fixture 中的旧产品名，导致检查必然失败。
- **根因 (Root cause)**：`scripts/check_tars_migration_paths.py` 只解析新增行内容，没有跟踪 unified diff 的当前文件路径，因此无法区分真实迁移内容与扫描器必须保存的匹配规则和测试样例。
- **为什么会写出 (Why)**：设计门禁时只考虑了“新增行/上下文行”边界，遗漏了规则实现本身也必须包含被禁止文本这一自引用边界。文本扫描器需要最小、可审计的路径级自豁免，不能用宽泛内容豁免。
- **修复 (Fix)**：解析 `+++ b/<path>` 文件边界，仅豁免门禁脚本及其测试两个精确路径；其他文件中的新增旧产品名仍返回失败。
- **测试拦截 (Test interception)**：`test_gate_allows_only_its_scanner_and_fixture_paths` 在同一 diff 中加入扫描器规则和普通文档违规，断言只放过前者；真实 `upstream/master..worktree` 门禁复验通过。
- **场景 (Scenario)**：从不含该门禁的上游基线检查新增门禁脚本，或更新扫描正则/测试 fixture 时触发。
- **关键词 (Keywords)**：migration path gate、self-match、unified diff、allowlist、product rename
