# FLORIDA KEYS RAW 

## DATA Structure
| deployment | t | t_increase | Vx | Vy | Vz | P | O2_S1 | O2_S2 | O2_S3 |


| Column | Type | Unit | Min | Max | Mean | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `deployment` | `str` | — | — | — | — | source filename; `YYYY_M_dayStart_dayEnd` |
| `t` | `float64` | **hours** | 10.00 | 56.00 | 27.57 | elapsed time |
| `t_increase` | `float64` | seconds | 0 | 72,000 | 32,476 | elapsed time within a deployment |
| `Vx` | `float64` | cm s⁻¹ | −49.816 | 30.791 | −3.255 | velocity |
| `Vy` | `float64` | cm s⁻¹ | −49.840 | 26.650 | −3.157 | velocity |
| `Vz` | `float64` | cm s⁻¹ | −49.662 | 8.854 | −0.687 | velocity |
| `P` | `float64` | kPa | 77.00 | 87.00 | 82.36 | gauge pressure |
| `O2_S1` | `float64` | µmol L⁻¹ | 232.676 | 271.396 | 249.987 | oxygen sensor 1 |
| `O2_S2` | `float64` | µmol L⁻¹ | 230.045 | 270.708 | 249.987 | oxygen sensor 2 |
| `O2_S3` | `float64` | µmol L⁻¹ | 229.973 | 270.722 | 249.987 | oxygen sensor 3 |


