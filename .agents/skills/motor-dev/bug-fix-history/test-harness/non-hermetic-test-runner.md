# [2026-08-28] 测试启动脚本修改环境、破坏参数边界并污染源码树

- **现象 (Symptom)**：`bash tests/run_tests.sh ...` 会把已安装的 `pytest-xdist` 误判为缺失并调用 `pip`，`-k "alpha or beta"` 等参数会被字符串展开拆散；每次运行都会把 `_pb2*.py` 写入源码目录，macOS 上还因 `sed -i` 语法不兼容失败；仅执行 `--help` 也会删除既有覆盖率文件。损坏的 Python 运行时会在 pytest 收集阶段以退出码 139 崩溃；ConfigWatcher 测试直接启动原生 FSEvents 时也可能报 `Cannot start fsevents stream` 后段错误。
- **根因 (Root cause)**：`tests/run_tests.sh` 混用 `python3`、`pip` 和 PATH 中的 `pytest`，用不存在的模块名 `pytest_xdist` 检查 xdist，把 pytest 参数保存在单个字符串中，并在参数解析前执行清理及在源码树内生成 proto；`scripts/generate_proto.sh` 固定原地输出且依赖 GNU `sed -i`；pre-commit 又直接引用 checkout 中不存在的 `./gitleaks`。测试环境使用的 Python 自身还会在 `import readline` 时崩溃；ConfigWatcher 测试则错误地把平台原生文件系统后端当作稳定的单元测试依赖。
- **为什么会写出 (Why)**：启动脚本被当作一次性本地便利工具，没有把解释器一致性、参数边界、只读工作树和跨平台行为作为测试基础设施契约；依赖缺失与解释器损坏也没有被区分。
- **修复 (Fix)**：`tests/run_tests.sh` 统一使用可显式指定的 `PYTHON_BIN`，只检查并报告依赖、绝不自动安装；使用 Bash 数组保留参数，检查 Python 运行时，按需清理覆盖率，并在自动清理的临时目录中生成 proto、pytest cache 和运行日志。`scripts/generate_proto.sh` 新增 `--output-dir`，用同一解释器生成代码，并以 Python 正则重写导入，保留默认原地输出的构建兼容行为。gitleaks hook 改用受控离线包装器，支持显式路径、仓根和 PATH，缺失时只报错。ConfigWatcher 单元测试统一通过低延迟 `PollingObserver` 启动真实 watcher，保留行为覆盖但移除平台后端依赖。
- **测试拦截 (Test interception)**：`tests/scripts/test_run_tests_harness.py` 覆盖 `--help` 零副作用、缺依赖 fail-fast、损坏运行时提前失败、参数边界、临时目录回收、pytest 退出码透传，以及 gitleaks 缺件失败和参数/退出码透传；`test_watcher_start_stop` 与文件修改用例通过 polling helper 覆盖启动、事件和停止行为；真实 etcd client、Scheduler 和 Router 定向测试验证临时 proto 导入及非 TARS 行为不变。
- **场景 (Scenario)**：开发机存在多个 Python、缺少测试依赖、Python 原生扩展损坏、在 macOS 运行测试、使用带空格的 `-k` 表达式，或要求测试前后工作树保持一致时。
- **关键词 (Keywords)**：run_tests.sh, PYTHON_BIN, pytest-xdist, FSEvents, PollingObserver
