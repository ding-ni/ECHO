"""Inventory genuine basin attributes and causal climate codes without Q/H reads."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from echo.geo.climate import historical_climate
from echo.utils import digest, read_json, utc_now, write_json

# Explicit independent-attribute selection: never feed the original full table.
# Scope 'u' = whole upstream watershed, NOT the local incremental subbasin.
INDEPENDENT_FIELDS = {
    "ele_mt_sav": "basin_elevation_mean", "ele_mt_smn": "basin_elevation_min",
    "ele_mt_smx": "basin_elevation_max", "slp_dg_uav": "upstream_slope_mean",
    "cly_pc_uav": "upstream_soil_clay_percent", "slt_pc_uav": "upstream_soil_silt_percent",
    "snd_pc_uav": "upstream_soil_sand_percent", "soc_th_uav": "upstream_soil_organic_carbon",
    "lit_cl_smj": "subbasin_lithology_majority_class", "kar_pc_use": "upstream_karst_percent",
    "for_pc_use": "upstream_forest_percent", "crp_pc_use": "upstream_cropland_percent",
    "pst_pc_use": "upstream_pasture_percent", "urb_pc_use": "upstream_urban_percent",
    "glc_cl_smj": "subbasin_landcover_majority_class", "gla_pc_use": "upstream_glacier_percent",
    "prm_pc_use": "upstream_permafrost_percent",
}
DIAGNOSTIC_FIELDS = {"snw_pc_syr": "annual_climatological_snow_cover_percent",
                     "snw_pc_smx": "maximum_month_climatological_snow_cover_percent"}


def make_inventory(h_index, attributes, output):
    h_index, attributes, output = Path(h_index), Path(attributes), Path(output)
    if output.exists():
        raise ValueError("环境资料版本已存在，请使用新目录")
    index = read_json(h_index)
    fields = {**INDEPENDENT_FIELDS, **DIAGNOSTIC_FIELDS}
    table = pd.read_csv(attributes, sep="\t", dtype={"STAID": str}, usecols=["STAID", *fields])
    table["STAID"] = table.STAID.str.replace(r"\.0$", "", regex=True).str.zfill(8)
    if table.STAID.duplicated().any():
        raise ValueError("属性表站号重复，不任意选择一条")
    table = table.set_index("STAID")
    output.mkdir(parents=True)
    records, tabular = [], []
    for row in index["stations"]:
        sid, role = row["station_id"], row["split"]
        if role not in {"train", "validation", "development_test"}:
            continue
        values = table.loc[sid] if sid in table.index else pd.Series(dtype=float)
        extracted = {name: (float(values[name]) if name in values and pd.notna(values[name])
                            and float(values[name]) > -999 else None) for name in fields}
        glacier = extracted["gla_pc_use"]
        if glacier is not None and not 0 <= glacier <= 100:
            raise ValueError("冰川覆盖字段超出0—100，先核验数据缩放")
        # For source training episodes starting in 2013, this fixed earlier climate
        # cutoff prevents even source episodes from using their query weather.
        cutoff = {"train": "2013-10-01", "validation": "2018-10-01",
                  "development_test": "2020-10-01"}[role]
        cutoff_ns = pd.Timestamp(cutoff, tz="UTC").value
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("已标准化气象与冻结水位视图哈希不一致")
        with np.load(row["path"], allow_pickle=False) as archive:
            # NPZ members are lazy: never access observed Q or H here.
            time_ns = archive["time_ns"]
            climate = historical_climate(archive["weather"], time_ns, cutoff_ns)
        path = output / f"{sid}_climate.npz"
        np.savez_compressed(path, climate_features=climate, climate_cutoff_ns=np.int64(cutoff_ns))
        record = {"station_id": sid, "split": role, "terminal_system_id": row["terminal_system_id"],
                  "independent_basin_attributes": {k: extracted[k] for k in INDEPENDENT_FIELDS},
                  "snow_climatology_diagnostic_only": {k: extracted[k] for k in DIAGNOSTIC_FIELDS},
                  "glacier_attribute_status": "unknown" if glacier is None else (
                      "positive_coverage_in_GLIMS_based_attribute" if glacier > 0 else "zero_in_archived_attribute"),
                  "climate_path": str(path.resolve()), "climate_sha256": digest(path),
                  "climate_cutoff_utc_exclusive": cutoff + "T00:00:00Z",
                  "climate_dimension": 64,
                  "usable_for_episode_origins_not_before_cutoff": True,
                  "incremental_node_attributes_complete": False,
                  "basin_attributes_may_be_broadcast_to_nodes": False,
                  "terrain_tokens_complete": False, "training_ready": False}
        records.append(record)
        tabular.append({"station_id": sid, "split": role, "terminal_system_id": row["terminal_system_id"],
                        **extracted, "climate_cutoff_utc": record["climate_cutoff_utc_exclusive"]})
    result = {"created_utc": utc_now(), "h_index_sha256": digest(h_index),
              "source_attributes_path": str(attributes.resolve()), "source_attributes_sha256": digest(attributes),
              "source_product": "CAMELSH station-aggregated HydroATLAS v1 attributes",
              "technical_documentation": "https://data.hydrosheds.org/file/technical-documentation/"
                                         "HydroATLAS_TechDoc_v10_1.pdf",
              "independent_field_meanings": INDEPENDENT_FIELDS, "diagnostic_field_meanings": DIAGNOSTIC_FIELDS,
              "snow_climatology_allowed_in_environment_g": False,
              "snow_climatology_reason": "long-term climatology period not verified against each forecast origin",
              "glacier_attribute_is_contemporary_field_survey": False,
              "HBV_has_snow_store_and_temperature_index_melt": True,
              "HBV_has_glacier_ice_mass_balance": False,
              "local_input_scope": "genuine_basin_summaries_and_preorigin_climate_not_complete_spatial_inputs",
              "stations": records, "target_Q_read": False, "H_values_read": False,
              "positive_glacier_attribute_stations": sum(
                  r["glacier_attribute_status"] == "positive_coverage_in_GLIMS_based_attribute" for r in records),
              "unknown_glacier_attribute_stations": sum(r["glacier_attribute_status"] == "unknown" for r in records),
              "protocol_training_ready": False}
    write_json(output / "index.json", result)
    pd.DataFrame(tabular).to_csv(output / "逐站独立属性与冰雪资料.csv", index=False, encoding="utf-8-sig")
    return {"stations": len(records), "climate_dimension": 64,
            "positive_glacier_attribute_stations": result["positive_glacier_attribute_stations"],
            "protocol_training_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-index", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(make_inventory(args.h_index, args.attributes, args.output))
