"""由显式下游关系追踪终端，拒绝把缺失链路当成终端。"""


def terminal_of(reach, downstream, outlets):
    seen = set()
    while reach not in outlets:
        if reach in seen:
            raise ValueError("河网有环")
        seen.add(reach)
        if reach not in downstream or downstream[reach] is None:
            raise ValueError(f"下游链路未知: {reach}")
        reach = downstream[reach]
    return str(reach)


def assign_terminals(station_reaches, downstream, outlets):
    return {station: terminal_of(reach, downstream, outlets) for station, reach in station_reaches.items()}
