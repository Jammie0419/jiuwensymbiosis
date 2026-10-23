# jiuwen_agx 包结构

```
jiuwen_agx/
  __init__.py              注册 capability / action / policy / skill
  actions.py               DIG / GET_TERRAIN / ACT_EXECUTE 的 ActionSpec
  contracts.py             零依赖的结果 TypedDict
  protocol.py              TerrainDriver 协议切片
  sim/                     通用仿真底座（与机型无关）
    config.py              SimMachineConfig：关节/单位/限位/底盘/地形
    driver.py              SimMachineDriver：校验 + backend 转发
    backend.py             Mock / InProcess / Remote 三种后端 + 注册表
    env.py                 SimMachineEnv：能力门控 + 观测
    api.py                 SimMachineApi：通用动作绑定
  agx_excavator/           挖掘机机型包（365 与 215F 共用）
    config.py              挖掘机专属字段（可达包络、dig 关键帧、IK/自然段、policy）
    work.py                dig 六段关键帧 + 可选自然段（纯 driver 函数）
    work_ik.py             3D 铲尖位姿 IK 工作循环（自由空间段走平滑轨迹）
    api.py                 dig / act_exec 绑定（dig_mode: ik|keyframe）
    env.py                 motion.excavator 能力
    session.py             make_builder + 资源键
    record.py              演示采集（npz）
  excavator_ik.py          4-DOF 平面 FK/IK（每机器 json 标定）
  trajectory.py            C¹ Hermite/PCHIP 关节轨迹生成（60 Hz 帧序列）
  policy.py                Policy 协议 + 注册表
  policy_act.py            lerobot ACT 适配
  policy_goal_act.py       goal-act-v2 适配
  machine_map.py           每机器 json 的原生单位 ↔ 策略弧度
  machine_fit.py           标定采样 → machine json
  cylinder_calibration.py  365 OpenPLX 几何扫描
```

## 扩展点

- **新仿真机型**：不新建适配器包。写一份 `configs/agx_excavator/agx_excavator.<name>.yaml`，
  再加一个场景工厂 `scripts/agx_<name>_scene.py`（`build()` 返回
  `sim/joints/units/machine_name/terrain_truth/load_mass_fn`；可选
  `terrain_truth_fn/tip_fn/surface_fn/base` 供 3D IK 与整机移动使用），在
  `agx_bridge_server.py` 的 `--machine` 注册表里加一项。动作层仍用 `dig`。
- **新动作/能力**：在 `actions.py` 声明 `ActionSpec`，在机型 `api.py` 用
  `@implements` 绑定；能力门控会自动生效。
- **新后端**：在 `sim/backend.py` 用 `@register_backend(name)` 注册，再在
  `sim/config.py` 的 `KNOWN_BACKENDS` 里放行。
- **新策略**：在 `policy.py` 用 `@register_policy(name)` 注册；配置里写 `policy.name` 即出现
  `act_exec`。
- **新机器标定**：按 `configs/machine_agx365.template.json` 的 schema 写
  `configs/machine_<machine>.json`，`MachineMap` / `act_exec` 自动读取。
