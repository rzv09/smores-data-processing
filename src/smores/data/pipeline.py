"""Run a site's pipeline: load -> harmonize -> 2-minute grid -> verify -> integrity -> write.

    python -m smores.data.pipeline                 # every site whose raw data is present
    python -m smores.data.pipeline biscayne_bay    # one or more named sites
    python -m smores.data.pipeline --no-write      # check only

Each stage has one job:

1. **load + harmonize** -- the site module (`florida_keys`, `biscayne_bay`,
   `santa_barbara`) reads its raw files and reshapes them to the shared schema at the
   native rate. Site-specific corrections live there and are documented there.
2. **grid** -- `resample.to_grid` median-bins onto `schema.SAMPLE_INTERVAL` (2 min).
3. **verify** -- `schema.validate`: columns, dtypes, enums, required non-nulls, order.
4. **integrity** -- `integrity.build_report`: the generic data checks plus the site's
   own. Any error-severity failure stops the run *before* anything is written, so a
   processed file on disk has always passed.
5. **write** -- `data/processed/<site>_o2_2min.parquet` and, next to it,
   `<site>_o2_2min_integrity.json` with every check result and the coverage table.
"""

import argparse
import sys
from dataclasses import dataclass

from smores import config
from smores.data import biscayne_bay, florida_keys, integrity, santa_barbara, schema


@dataclass(frozen=True)
class Site:
    module: object
    raw_path: object
    out_path: object

    @property
    def site_id(self):
        return self.module.SITE_ID

    @property
    def report_path(self):
        return self.out_path.with_name(self.out_path.stem + "_integrity.json")

    def available(self):
        return self.raw_path.exists()


SITES = {
    "florida_keys": Site(florida_keys, config.FLORIDA_KEYS_CSV, config.FLORIDA_KEYS_2MIN_PARQUET),
    "biscayne_bay": Site(
        biscayne_bay, config.BISCAYNE_HARVARD_DIR, config.BISCAYNE_BAY_2MIN_PARQUET
    ),
    "santa_barbara": Site(
        santa_barbara, config.SANTA_BARBARA_CSV, config.SANTA_BARBARA_2MIN_PARQUET
    ),
}


def run(site_id, write=True, raw_path=None, out_path=None, verbose=True):
    """Run one site end to end. Returns (frame, report); raises on integrity failure."""
    site = SITES[site_id]
    grid, site_checks = site.module.pipeline(raw_path or site.raw_path, verbose=verbose)

    frame = schema.validate(grid.drop(columns="n_obs"))
    report = integrity.build_report(grid, site_id, site_checks)
    if verbose:
        print(report)
    report.raise_for_failures()

    if write:
        destination = out_path or site.out_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, compression="snappy", index=False)
        report.write(destination.with_name(destination.stem + "_integrity.json"))
        if verbose:
            print(f"wrote {len(frame):,} rows to {destination}")
    return frame, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sites", nargs="*", help=f"any of {list(SITES)}; default: all available")
    parser.add_argument("--no-write", action="store_true", help="run every check, write nothing")
    args = parser.parse_args(argv)

    unknown = sorted(set(args.sites) - set(SITES))
    if unknown:
        parser.error(f"unknown site(s) {unknown}; choose from {list(SITES)}")

    names = args.sites or [name for name, site in SITES.items() if site.available()]
    failed = []
    for name in names:
        if not SITES[name].available():
            print(f"[{name}] raw data not found at {SITES[name].raw_path}; skipped")
            continue
        try:
            run(name, write=not args.no_write)
        except integrity.IntegrityError as error:
            print(error, file=sys.stderr)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
