"""严格白名单；统计量仅由来源训练组拟合。"""

import numpy as np

ALLOWED = {
    "area_km2",
    "slope_degrees",
    "relief_m",
    "latitude",
    "longitude",
    "elevation_m",
    "forest_fraction",
    "soil_sand_fraction",
    "soil_clay_fraction",
    "geology_permeability",
}


def validate_features(names):
    invalid = set(names) - ALLOWED
    if invalid:
        raise ValueError(f"未批准的环境特征: {sorted(invalid)}")


class FeatureScaler:
    def fit(self, x):
        x = np.asarray(x, float)
        self.center = np.nanmedian(x, axis=0)
        self.center = np.nan_to_num(self.center)
        self.scale = np.nanstd(x, axis=0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-6), self.scale, 1.0)
        return self

    def transform(self, x):
        x = np.asarray(x, float)
        missing = ~np.isfinite(x)
        return np.concatenate(
            [(np.where(missing, self.center, x) - self.center) / self.scale, missing.astype(float)], axis=-1
        ).astype("float32")
