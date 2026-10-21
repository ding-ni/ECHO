"""正线性库精确步积分；输入为步平均流率，储量与时间单位必须一致。"""

import torch


def reservoir_step(storage, inflow, tau, dt=1.0):
    if torch.any(tau <= 0):
        raise ValueError("时间尺度必须为正")
    decay = torch.exp(-dt / tau)
    nxt = decay * storage + tau * (-torch.expm1(-dt / tau)) * inflow
    out = (storage + dt * inflow - nxt) / dt
    return nxt, out


def route_network(volume, lateral, tau_seconds, parents, dt=3600.0):
    """节点须拓扑排序；只接受增量侧向流，不重复累计面积。"""
    if len(parents) != volume.shape[-1]:
        raise ValueError("河网节点数不匹配")
    inputs = list(lateral.unbind(-1))
    volumes, outputs = [], []
    for j, parent in enumerate(parents):
        if parent != -1 and parent <= j:
            raise ValueError("河网不是上游到下游的拓扑顺序")
        v, out = reservoir_step(volume[..., j], inputs[j], tau_seconds[..., j], dt)
        volumes.append(v)
        outputs.append(out)
        if parent != -1:
            if parent >= len(parents):
                raise ValueError("非法下游节点")
            inputs[parent] = inputs[parent] + out
    return torch.stack(volumes, -1), torch.stack(outputs, -1)
